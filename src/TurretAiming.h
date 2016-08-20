#pragma once

#include <opencv2\opencv.hpp>

#include <unordered_map>
#include <memory>

namespace TurretAiming {

	struct SimulationElement {
		virtual void display(cv::Mat& im) const = 0;
		virtual void update(
			float dt // time since last update
		) = 0;
	};

	struct Ball : SimulationElement {
		float x, y;
		float sX, sY;
		float radius = 0.02f;
		void init(float x, float y, float sX, float sY) {
			this->x = x; this->y = y;
			this->sX = sX; this->sY = sY;
		}
		void display(cv::Mat& im, cv::Scalar color) const {
			int w = im.size().width, h = im.size().height;
			double s = sqrt(w*h);
			cv::circle(
				im,
				{ int(x * w), int(y * h) },
				int( s * this->radius ),
				color,
				CV_FILLED
			);
		}
		void display(cv::Mat& im) const { this->display(im, { 255.0, 255.0, 255.0 }); }
		void update( float dt = 1 ) {
			x += dt * sX; y += dt * sY;
		}
	};

	struct Target : Ball {
		void display(cv::Mat& im) const { this->Ball::display(im, { 64.0, 0.0, 256.0 }); }
		void update(float dt = 1) {
			this->Ball::update(dt);
			if (x > 1) { sX = -abs(sX); }
			if (x < 0) { sX = abs(sX); }
			if (y > 1) { sY = -abs(sY); }
			if (y < 0) { sY = abs(sY); }
		}
	};

	struct Bullets : SimulationElement {

		float numberMiss = 0, numberHit = 0;

		struct Bullet : Ball {
			bool live = true, hit = false;
			Bullet() { radius = 0.01f; }
			void display(cv::Mat& im) const { this->Ball::display(im, { 112.0, 132.0, 128.0 }); }
			void update(float dt) {
				this->Ball::update(dt);
				if (
					x > 1 + radius ||
					x < 0 - radius ||
					y > 1 + radius ||
					x < 0 - radius
				) {
					this->live = false;
				}
			}
			void detectHit(const Target& target) {
				float dx = target.x - this->x;
				float dy = target.y - this->y;
				float dr = target.radius + this->radius;
				if (dx*dx + dy*dy < dr * dr) {
					this->live = false;
					this->hit = true;
				}
			}
		};

		int maxBullets = 1000;
		int nextId;
		std::unordered_map<int,Bullet> bullets;

		void addBullet(Bullet b) {
			bullets[nextId] = b;
			nextId = (nextId + 1) % maxBullets;
			if (bullets.size() == maxBullets) {
				std::cout << "Warning : just passed the limit of "
					<< maxBullets << "bullets. Starting to deleted some" << std::endl;
			}
		}

		void display(cv::Mat& im) const {
			for (const auto& b : bullets) {
				b.second.display(im);
			}
		}

		void detectHits(const Target& target) {
			for (auto& b : bullets) {
				b.second.detectHit(target);
			}
		}

		void update(float dt) {
			std::vector<int> toDelete;
			for (auto& b : bullets) {
				b.second.update(dt);
				if (!b.second.live) {
					toDelete.push_back(b.first);
					if (b.second.hit) {
						numberHit++;
					} else {
						numberMiss++;
					}
				}
			}
			// deleting dead bullets
			for (int b : toDelete) {
				bullets.erase(b);
			}
			numberHit *= pow(0.99, dt);
			numberMiss *= pow(0.99, dt);
		}

		float ratioHit() const {
			return this->numberHit / std::max(1.0f, this->numberMiss + this->numberHit);
		}
	};

	struct Turret : SimulationElement {

		float x = 0.5f, y = 0.5f; // pixel pos in [0;1]
		float angle = 0.123f; // in radians

		Bullets* bullets;

		void display(cv::Mat& im, cv::Scalar color) const {

			int w = im.size().width, h = im.size().height;
			double s = sqrt(w*h);
			cv::Point center = { int(x*w), int(y*h) };
			cv::circle(im, center, int(0.03*s), color);
			cv::line(
				im,
				center,
				{
					int(x*w + 0.05*s*cos(this->angle)),
					int(y*h + 0.05*s*sin(this->angle))
				},
				color
			);
		}
		void display(cv::Mat& im) const { this->display(im, { 64.0, 128.0, 255.0 }); }

		float firePeriod = 3.17f;
		float countdown = firePeriod;
		float bulletSpeed = 0.01f;

		virtual void aim(const Target& target) = 0;

		void update(float dt) {
			this->countdown -= dt;
			while (this->countdown < 0) {
				this->countdown += firePeriod;
				Bullets::Bullet b;
				b.init(this->x, this->y, this->bulletSpeed * cos(this->angle), this->bulletSpeed * sin(this->angle));
				bullets->addBullet(b);
			}
		}
	};

	struct DummyTurret : Turret {
		void aim(const Target& target) { this->angle += 0.132f; }
	};

	struct DirectTurret : Turret {
		void aim(const Target& target) {
			this->angle = atan2(target.y - this->y, target.x - this->x);
		}
	};

	struct SpeedTurret : Turret {
		void aim(const Target& target) {
			float dx = this->x - target.x;
			float dy = this->y - target.y;
			float d = sqrtf(dx*dx + dy*dy);
			float x = target.x + target.sX * d / this->bulletSpeed;
			float y = target.y + target.sY * d / this->bulletSpeed;
			/*while (x > 1 || x < 0 || y > 1 || y < 0) {
				if (x > 1) { x = 1 - x; }
				if (x < 0) { x = -x; }
				if (y > 1) { y = 1 - y; }
				if (y < 0) { y = -y; }
			}*/
			this->angle = atan2(y - this->y, x - this->x);
		}
	};

	struct InertiaTurret : SpeedTurret {
		float targetAngle;
		void update(float dt) {
			float d1 = this->targetAngle - this->angle;
			float d2 = this->targetAngle + 2 * 3.1416 - this->angle; // TODO : BUG ?
			float dir = (abs(d1) < abs(d2) ? signbit(d1) : signbit(d2)) ? -1.0 : 1.0;
			this->angle += 0.1 * std::min(1.0f, abs(d1) < abs(d2) ? abs(d1) : abs(d2) )* dt * dir;
			SpeedTurret::update(dt);
		}
		void aim(const Target& target) {
			float prevAngle = this->angle;
			SpeedTurret::aim(target);
			this->targetAngle = this->angle;
			this->angle = prevAngle;
		}
	};

	struct Simulation : SimulationElement {

		std::shared_ptr<Turret> turret;
		Target target;
		Bullets bullets;

		int subSteps = 1;
		float speed = 1;

		std::vector<SimulationElement*> elements;

		Simulation()
			: turret(new InertiaTurret())
		{
			turret->bullets = &this->bullets;
			target.init( 0.7f, 0.2f, 0.01234f, 0.018f );
			this->elements.push_back(this->turret.get());
			this->elements.push_back(&this->target);
			this->elements.push_back(&this->bullets);
		}

		void update( float dt = 1 ) {
			for (SimulationElement* el : this->elements) {
				el->update( dt );
			}
			this->bullets.detectHits(this->target);
		}

		void display(cv::Mat& im) const {
			for (const SimulationElement* el : this->elements) {
				el->display(im);
			}
			std::stringstream ss;
			ss << int(this->bullets.ratioHit()*100.0) << " % hits";
			cv::putText(im, ss.str(), { 10, 20 }, CV_FONT_HERSHEY_COMPLEX, 0.5, { 128.0, 128.0, 128.0 });
		}

		void run() {
			cv::Mat im(cv::Size(512, 512), CV_8UC3); im = 0;
			while (true) {
				for (int i = 0; i < subSteps * speed; i++)
				{
					//im -= 1.0 * cv::Scalar{ 1.0, 1.0, 1.0 }; im *= 256.0 / (256.0 - 1.0);
					im *= pow(0.95,1.0/subSteps);
					//im = 0;
					this->display(im);
					this->turret->aim(this->target);
					this->update(1.0f/subSteps);
				}
				cv::imshow("Simulation", im);
				if (cv::waitKey(6) == 27) { return; }
			}
		}
	};
}