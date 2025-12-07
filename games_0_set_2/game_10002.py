import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:45:31.699475
# Source Brief: brief_00002.md
# Brief Index: 2
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Navigate a spaceship through a hazardous asteroid field. Match your weapon's color to the asteroids to destroy them, and terraform special asteroids for a boost."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire. Hold shift and use ↑/↓ to change ship size. Tap shift to terraform nearby asteroids."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_SHIP = (230, 255, 255)
        self.COLOR_SHIP_BOOST = (255, 255, 255)
        self.COLOR_SHIP_GLOW = (100, 180, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.ASTEROID_COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 120, 255),
            "yellow": (255, 255, 80)
        }
        self.COLOR_NAMES = list(self.ASTEROID_COLORS.keys())[:3] # red, green, blue

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_ui = pygame.font.SysFont("Consolas", 18)

        # Persistent state (survives reset)
        self.max_ship_speed_multiplier = 1.0

        # Initialize state variables
        self.ship_pos = None
        self.ship_vel = None
        self.ship_size = None
        self.target_ship_size = None
        self.asteroids = []
        self.bullets = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.color_pattern = []
        self.shoot_cooldown = 0
        self.terraform_cooldown = 0
        self.boost_timer = 0
        self.asteroid_pulsation_speed = 0.0
        self.last_shift_held = False
        self.reward_milestones = {}

        self._generate_stars()
        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ship_pos = pygame.Vector2(self.WIDTH / 4, self.HEIGHT / 2)
        self.ship_vel = pygame.Vector2(0, 0)
        self.ship_size = 20
        self.target_ship_size = 20

        self.asteroids = []
        for _ in range(15):
            self._spawn_asteroid(random_x=True)

        self.bullets = []
        self.particles = []

        self.color_pattern = [self.np_random.choice(self.COLOR_NAMES)]
        self.shoot_cooldown = 0
        self.terraform_cooldown = 0
        self.boost_timer = 0
        self.last_shift_held = False
        self.asteroid_pulsation_speed = 0.1
        
        self.reward_milestones = {1000: False, 2500: False, 5000: False}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        self._handle_input(action)
        self._update_game_state()
        
        reward += self._handle_collisions()
        
        self._update_difficulty()
        self._cleanup_entities()
        if self.np_random.random() < 0.05:
            self._spawn_asteroid()

        self.steps += 1
        reward += self._calculate_reward()
        terminated = self._check_termination()

        if terminated and not self.game_over:
            # Reached max steps, not a crash
            pass
        elif terminated and self.game_over:
            # Crashed into asteroid
            reward -= 10 # Penalty for crashing
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Sizing ---
        # Shift + Up/Down for size change
        if shift_held:
            if movement == 1: # Up
                self.target_ship_size += 1
            elif movement == 2: # Down
                self.target_ship_size -= 1
        self.target_ship_size = np.clip(self.target_ship_size, 10, 40)

        # --- Movement ---
        # Normal movement if shift is not held
        move_vec = pygame.Vector2(0, 0)
        if not shift_held:
            if movement == 1: # Up
                move_vec.y = -1
            elif movement == 2: # Down
                move_vec.y = 1
            elif movement == 3: # Left
                move_vec.x = -1
            elif movement == 4: # Right
                move_vec.x = 1
        
        # Apply acceleration
        if move_vec.length() > 0:
            self.ship_vel += move_vec.normalize() * 0.8
        
        # --- Shooting ---
        if space_held and self.shoot_cooldown == 0:
            # SFX: Laser shoot
            for color in self.color_pattern:
                self.bullets.append({
                    "pos": self.ship_pos.copy(),
                    "vel": pygame.Vector2(15, 0),
                    "color_name": color,
                    "color_rgb": self.ASTEROID_COLORS[color]
                })
            self.shoot_cooldown = 8 # Cooldown in frames

        # --- Terraforming ---
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed and self.terraform_cooldown == 0:
            self._attempt_terraform()
        
        self.last_shift_held = shift_held

    def _attempt_terraform(self):
        target_asteroid = None
        min_dist = 150 # Max terraform range

        for asteroid in self.asteroids:
            if asteroid["size"] < 20 and asteroid["color_name"] != "yellow":
                dist = self.ship_pos.distance_to(asteroid["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target_asteroid = asteroid
        
        if target_asteroid:
            # SFX: Terraform charge
            target_asteroid["color_name"] = "yellow"
            target_asteroid["color_rgb"] = self.ASTEROID_COLORS["yellow"]
            target_asteroid["target_size"] = self.np_random.uniform(40, 50)
            self.terraform_cooldown = 60 # Cooldown in frames
            self._create_effect(target_asteroid["pos"], self.ASTEROID_COLORS["yellow"], 20, 1.5)

    def _update_game_state(self):
        # Cooldowns
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.terraform_cooldown > 0: self.terraform_cooldown -= 1
        
        # Boost
        if self.boost_timer > 0: self.boost_timer -= 1
        speed_multiplier = self.max_ship_speed_multiplier if self.boost_timer > 0 else 1.0

        # Ship
        self.ship_vel *= 0.9 # Friction
        if self.ship_vel.length() < 0.1: self.ship_vel = pygame.Vector2(0, 0)
        self.ship_pos += self.ship_vel
        self.ship_pos.x = np.clip(self.ship_pos.x, 0, self.WIDTH)
        self.ship_pos.y = np.clip(self.ship_pos.y, 0, self.HEIGHT)
        self.ship_size = self._lerp(self.ship_size, self.target_ship_size, 0.1)

        # Bullets
        for bullet in self.bullets:
            bullet["pos"] += bullet["vel"]

        # Asteroids
        for asteroid in self.asteroids:
            asteroid["pos"].x -= (asteroid["speed"] * speed_multiplier)
            
            # Pulsation
            if asteroid["size"] >= asteroid["target_size"]: asteroid["growth_dir"] = -1
            if asteroid["size"] <= asteroid["min_size"]: asteroid["growth_dir"] = 1
            
            pulse_amount = self.asteroid_pulsation_speed * asteroid["growth_dir"]
            asteroid["size"] += pulse_amount
            
            if abs(asteroid["size"] - asteroid["target_size"]) < 0.2 and asteroid["growth_dir"] == 1:
                asteroid["target_size"] = self.np_random.uniform(asteroid["base_size"] * 0.8, asteroid["base_size"] * 1.2)
            if abs(asteroid["size"] - asteroid["min_size"]) < 0.2 and asteroid["growth_dir"] == -1:
                asteroid["min_size"] = self.np_random.uniform(asteroid["base_size"] * 0.3, asteroid["base_size"] * 0.7)


        # Particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["size"] = max(0, p["size"] - 0.1)

        # Stars
        for star in self.stars:
            star["pos"].x -= star["speed"] * speed_multiplier
            if star["pos"].x < 0:
                star["pos"].x = self.WIDTH
                star["pos"].y = self.np_random.uniform(0, self.HEIGHT)

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Asteroids
        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                if bullet["pos"].distance_to(asteroid["pos"]) < asteroid["size"]:
                    if bullet["color_name"] == asteroid["color_name"]:
                        # SFX: Asteroid explosion
                        self._create_effect(asteroid["pos"], asteroid["color_rgb"], int(asteroid["size"]), 2.0)
                        self.asteroids.remove(asteroid)
                        if bullet in self.bullets: self.bullets.remove(bullet)
                        self.score += 1
                        reward += 1
                        break # Next bullet
                    elif asteroid["color_name"] == "yellow":
                        # SFX: Boost pickup
                        self._create_effect(asteroid["pos"], asteroid["color_rgb"], int(asteroid["size"] * 1.5), 3.0)
                        self.asteroids.remove(asteroid)
                        if bullet in self.bullets: self.bullets.remove(bullet)
                        self.score += 5
                        reward += 5
                        self.boost_timer = 150 # 5 seconds at 30fps
                        break

        # Ship vs Asteroids
        for asteroid in self.asteroids:
            if self.ship_pos.distance_to(asteroid["pos"]) < (self.ship_size * 0.8 + asteroid["size"] * 0.8):
                # SFX: Player explosion
                self.game_over = True
                self._create_effect(self.ship_pos, self.COLOR_SHIP, 30, 4.0)
                break
        
        return reward

    def _update_difficulty(self):
        # Asteroid pulsation speed
        if self.steps > 0 and self.steps % 500 == 0:
            self.asteroid_pulsation_speed = min(0.5, self.asteroid_pulsation_speed + 0.02)
        
        # Color pattern complexity
        if self.score > 0 and (self.score // 1000) > (len(self.color_pattern) - 1) and len(self.color_pattern) < 3:
            new_color = self.np_random.choice([c for c in self.COLOR_NAMES if c not in self.color_pattern])
            self.color_pattern.append(new_color)

        # Ship speed upgrade
        if self.score > 0 and (self.score // 2000) > (self.max_ship_speed_multiplier - 1.0) / 0.1:
            self.max_ship_speed_multiplier = min(3.0, self.max_ship_speed_multiplier + 0.1)

    def _cleanup_entities(self):
        self.bullets = [b for b in self.bullets if 0 < b["pos"].x < self.WIDTH]
        self.asteroids = [a for a in self.asteroids if a["pos"].x > -a["size"]]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
    
    def _spawn_asteroid(self, random_x=False):
        size = self.np_random.uniform(15, 50)
        pos = pygame.Vector2(
            self.WIDTH + size if not random_x else self.np_random.uniform(self.WIDTH/2, self.WIDTH),
            self.np_random.uniform(size, self.HEIGHT - size)
        )
        color_name = self.np_random.choice(self.COLOR_NAMES + ["yellow"], p=[0.35, 0.35, 0.2, 0.1])
        
        self.asteroids.append({
            "pos": pos,
            "size": size,
            "base_size": size,
            "target_size": size * self.np_random.uniform(0.8, 1.2),
            "min_size": size * self.np_random.uniform(0.3, 0.7),
            "growth_dir": 1,
            "speed": self.np_random.uniform(1.0, 3.0),
            "color_name": color_name,
            "color_rgb": self.ASTEROID_COLORS[color_name],
            "vertices": self._create_asteroid_shape(size)
        })

    def _calculate_reward(self):
        reward = 0.1 # Survival reward
        
        for milestone, achieved in self.reward_milestones.items():
            if not achieved and self.steps >= milestone:
                if milestone == 1000: reward += 10
                elif milestone == 2500: reward += 25
                elif milestone == 5000: reward += 50
                self.reward_milestones[milestone] = True
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_background()
        self._render_asteroids()
        self._render_bullets()
        if not self.game_over:
            self._render_ship()
        self._render_particles()

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star["color"], (int(star["pos"].x), int(star["pos"].y)), int(star["size"]))
    
    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(v * asteroid["size"] + asteroid["pos"]) for v in asteroid["vertices"]]
            points_int = [(int(p.x), int(p.y)) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, points_int, asteroid["color_rgb"])
            pygame.gfxdraw.filled_polygon(self.screen, points_int, asteroid["color_rgb"])

    def _render_bullets(self):
        for bullet in self.bullets:
            pos = (int(bullet["pos"].x), int(bullet["pos"].y))
            pygame.draw.circle(self.screen, bullet["color_rgb"], pos, 5)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, bullet["color_rgb"])

    def _render_ship(self):
        ship_color = self.COLOR_SHIP_BOOST if self.boost_timer > 0 else self.COLOR_SHIP
        
        # Boost trail
        if self.boost_timer > 0:
            for i in range(5):
                alpha = 150 - i * 30
                color = (*self.COLOR_SHIP_BOOST, alpha)
                s = pygame.Surface((self.ship_size*2, self.ship_size*2), pygame.SRCALPHA)
                trail_pos = self.ship_pos - self.ship_vel * i * 0.5
                p1 = (self.ship_size, self.ship_size - self.ship_size)
                p2 = (self.ship_size - self.ship_size * 0.7, self.ship_size + self.ship_size * 0.7)
                p3 = (self.ship_size + self.ship_size * 0.7, self.ship_size + self.ship_size * 0.7)
                pygame.draw.polygon(s, color, [p1, p2, p3])
                self.screen.blit(s, (trail_pos.x - self.ship_size, trail_pos.y - self.ship_size))

        # Glow effect
        glow_size = int(self.ship_size * 2.5)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        glow_alpha = 80 + (self.boost_timer % 10) * 5
        pygame.draw.circle(glow_surface, (*self.COLOR_SHIP_GLOW, glow_alpha), (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surface, (int(self.ship_pos.x - glow_size/2), int(self.ship_pos.y - glow_size/2)))

        # Ship body
        p1 = (self.ship_pos.x, self.ship_pos.y - self.ship_size)
        p2 = (self.ship_pos.x - self.ship_size * 0.7, self.ship_pos.y + self.ship_size * 0.7)
        p3 = (self.ship_pos.x + self.ship_size * 0.7, self.ship_pos.y + self.ship_size * 0.7)
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, ship_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, ship_color)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Speed Multiplier
        speed_text = self.font_main.render(f"SPEED: {self.max_ship_speed_multiplier:.1f}x", True, self.COLOR_UI_TEXT)
        text_rect = speed_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(speed_text, text_rect)
        if self.boost_timer > 0:
            boost_indicator = self.font_ui.render("BOOST ACTIVE", True, self.ASTEROID_COLORS["yellow"])
            ind_rect = boost_indicator.get_rect(topright=(self.WIDTH - 10, 35))
            self.screen.blit(boost_indicator, ind_rect)

        # Color Pattern
        pattern_y = self.HEIGHT - 30
        total_width = len(self.color_pattern) * 25 - 5
        start_x = (self.WIDTH - total_width) / 2
        for i, color_name in enumerate(self.color_pattern):
            color_rgb = self.ASTEROID_COLORS[color_name]
            pygame.draw.rect(self.screen, color_rgb, (start_x + i * 25, pattern_y, 20, 20), border_radius=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ship_size": self.ship_size,
            "boost_timer": self.boost_timer,
            "max_speed_multiplier": self.max_ship_speed_multiplier
        }

    # Helper methods
    def _lerp(self, start, end, amt):
        return start + (end - start) * amt

    def _create_asteroid_shape(self, radius):
        num_vertices = self.np_random.integers(7, 12)
        vertices = []
        for i in range(num_vertices):
            angle = (2 * math.pi / num_vertices) * i
            dist = self.np_random.uniform(0.7, 1.1)
            vec = pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
            vertices.append(vec)
        return vertices

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            depth = self.np_random.uniform(0.1, 1.0)
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "speed": 0.5 * depth,
                "size": 1.0 * depth,
                "color": (100 + 100 * depth, 100 + 100 * depth, 120 + 100 * depth)
            })

    def _create_effect(self, pos, color, num_particles, speed_mult):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "size": self.np_random.uniform(2, 6),
                "color": color
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block is for manual play and debugging.
    # It will not be executed in the evaluation environment.
    # Set a non-dummy video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Controls ---
    # ARROWS: Move (if Shift not held)
    # SHIFT + UP/DOWN: Grow/Shrink ship
    # SHIFT (press): Terraform nearest small asteroid
    # SPACE: Shoot
    # Q: Quit
    
    # Create a display surface for rendering
    pygame.display.init()
    display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("AstroSizer")

    while not terminated:
        # Map pygame keys to gymnasium action
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if not keys[pygame.K_LSHIFT] and not keys[pygame.K_RSHIFT]:
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
        else: # Shift is held, up/down is for sizing
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surf.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                terminated = True
                
        env.clock.tick(env.FPS)
        
    env.close()
    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")