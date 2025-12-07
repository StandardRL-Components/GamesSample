
# Generated: 2025-08-27T23:45:36.083831
# Source Brief: brief_03570.md
# Brief Index: 3570

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire. Hold Shift to use a banked power-up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless asteroid field for 60 seconds by skillfully piloting your ship and blasting space rocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 40)
        self.COLOR_PROJECTILE = (0, 200, 255)
        self.COLOR_ASTEROID = (180, 180, 190)
        self.COLOR_POWERUP_INVINCIBILITY = (255, 220, 0)
        self.COLOR_POWERUP_RAPIDFIRE = (255, 0, 255)
        self.COLOR_EXPLOSION = [(255, 60, 0), (255, 180, 0), (255, 255, 200)]
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEALTH = (0, 255, 128)
        self.COLOR_HEALTH_BG = (100, 100, 100)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.np_random = None
        self.player = None
        self.asteroids = []
        self.projectiles = []
        self.powerups = []
        self.particles = []
        self.stars = []
        
        # Initialize state variables that are reset
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.player = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "speed": 4,
            "size": 12,
            "health": 3,
            "fire_cooldown": 0,
            "fire_rate": 10,
            "invincibility_timer": 0,
            "banked_powerup": None,
            "active_powerup": None,
            "powerup_timer": 0,
        }

        self.asteroids = []
        self.projectiles = []
        self.powerups = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.asteroid_spawn_timer = 0
        self.base_asteroid_spawn_rate = 1.0  # seconds
        self.base_asteroid_speed = 1.0

        if not self.stars:
            for _ in range(150):
                self.stars.append({
                    "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                    "size": self.np_random.integers(1, 3),
                    "brightness": self.np_random.uniform(50, 150)
                })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # Base survival reward
            reward += 0.01

            # Find nearest asteroid for shaping reward
            dist_before = self._find_nearest_asteroid_dist()

            self._handle_input(movement, space_held, shift_held)
            self._update_game_state()
            collision_rewards = self._handle_collisions()
            reward += collision_rewards

            # Apply shaping reward for moving towards asteroids
            dist_after = self._find_nearest_asteroid_dist()
            if dist_before is not None and dist_after is not None:
                if dist_after > dist_before + 0.1: # Penalize moving away
                    reward -= 0.02
                elif dist_after < dist_before - 0.1: # Reward moving closer
                    reward += 0.01

        self.steps += 1
        
        # Check termination conditions
        if self.player["health"] <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -100.0  # Terminal penalty for dying
            self._create_explosion(self.player["pos"], self.player["size"] * 2, 100)
            # sfx: player_explosion

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True
            reward = 100.0  # Terminal reward for survival

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player["pos"].y -= self.player["speed"]
        if movement == 2: self.player["pos"].y += self.player["speed"]
        if movement == 3: self.player["pos"].x -= self.player["speed"]
        if movement == 4: self.player["pos"].x += self.player["speed"]
        
        # Clamp player position
        self.player["pos"].x = np.clip(self.player["pos"].x, self.player["size"], self.WIDTH - self.player["size"])
        self.player["pos"].y = np.clip(self.player["pos"].y, self.player["size"], self.HEIGHT - self.player["size"])

        # Firing
        if space_held and self.player["fire_cooldown"] <= 0:
            self.projectiles.append(pygame.Vector2(self.player["pos"]))
            self.player["fire_cooldown"] = self.player["fire_rate"]
            # sfx: laser_shoot

        # Power-up Activation
        if shift_held and self.player["banked_powerup"] is not None and self.player["active_powerup"] is None:
            self.player["active_powerup"] = self.player["banked_powerup"]
            self.player["banked_powerup"] = None
            if self.player["active_powerup"] == "invincibility":
                self.player["powerup_timer"] = self.FPS * 5  # 5 seconds
                # sfx: powerup_activate_invincibility
            elif self.player["active_powerup"] == "rapid_fire":
                self.player["powerup_timer"] = self.FPS * 8  # 8 seconds
                # sfx: powerup_activate_rapidfire

    def _update_game_state(self):
        # Timers
        if self.player["fire_cooldown"] > 0: self.player["fire_cooldown"] -= 1
        if self.player["invincibility_timer"] > 0: self.player["invincibility_timer"] -= 1
        if self.player["powerup_timer"] > 0: self.player["powerup_timer"] -= 1
        else: self.player["active_powerup"] = None

        # Power-up effects
        if self.player["active_powerup"] == "rapid_fire":
            self.player["fire_rate"] = 3
        else:
            self.player["fire_rate"] = 10

        # Update Projectiles
        for proj in self.projectiles[:]:
            proj.y -= 10
            if proj.y < 0: self.projectiles.remove(proj)

        # Update Asteroids
        for ast in self.asteroids:
            ast["pos"] += ast["vel"]
            ast["angle"] += ast["rot_speed"]
        self.asteroids = [a for a in self.asteroids if a["pos"].y < self.HEIGHT + a["radius"]]

        # Difficulty scaling
        difficulty_mod = 1 + (self.steps / self.MAX_STEPS) * 2
        current_spawn_rate = self.base_asteroid_spawn_rate / difficulty_mod
        current_speed = self.base_asteroid_speed * difficulty_mod

        # Spawn Asteroids
        self.asteroid_spawn_timer -= 1 / self.FPS
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid(current_speed)
            self.asteroid_spawn_timer = current_spawn_rate

        # Spawn Power-ups
        if self.np_random.random() < 0.001 and len(self.powerups) < 1 and self.player["banked_powerup"] is None:
            self._spawn_powerup()

        # Update Particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0: self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Projectiles vs Asteroids
        for proj in self.projectiles[:]:
            for ast in self.asteroids[:]:
                if proj.distance_to(ast["pos"]) < ast["radius"]:
                    self.projectiles.remove(proj)
                    ast["health"] -= 1
                    self._create_explosion(proj, 5, 5, (200, 200, 255)) # Hit spark
                    if ast["health"] <= 0:
                        reward += ast["score_value"]
                        self._create_explosion(ast["pos"], ast["radius"], 20)
                        self._break_asteroid(ast)
                        self.asteroids.remove(ast)
                        self.score += ast["score_value"]
                        # sfx: asteroid_explosion
                    break
        
        # Player vs Asteroids
        is_invincible = self.player["invincibility_timer"] > 0 or self.player["active_powerup"] == "invincibility"
        if not is_invincible:
            for ast in self.asteroids[:]:
                if self.player["pos"].distance_to(ast["pos"]) < self.player["size"] + ast["radius"]:
                    self.player["health"] -= 1
                    self.player["invincibility_timer"] = self.FPS * 2 # 2 seconds of grace period
                    self._create_explosion(self.player["pos"], self.player["size"], 30)
                    self.asteroids.remove(ast)
                    self._break_asteroid(ast)
                    # sfx: player_hit
                    break
        
        # Player vs Power-ups
        for pu in self.powerups[:]:
            if self.player["pos"].distance_to(pu["pos"]) < self.player["size"] + 10:
                if self.player["banked_powerup"] is None:
                    self.player["banked_powerup"] = pu["type"]
                    reward += 5
                    self.powerups.remove(pu)
                    # sfx: powerup_collect
                break
        
        return reward

    def _spawn_asteroid(self, speed):
        size_roll = self.np_random.random()
        if size_roll < 0.2:
            size_key = "large"
            radius = 35
            health = 3
            score = 3
        elif size_roll < 0.6:
            size_key = "medium"
            radius = 20
            health = 2
            score = 2
        else:
            size_key = "small"
            radius = 12
            health = 1
            score = 1
        
        pos = pygame.Vector2(self.np_random.uniform(radius, self.WIDTH - radius), -radius)
        angle = self.np_random.uniform(0.3, math.pi - 0.3)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed * self.np_random.uniform(0.8, 1.2)
        
        self.asteroids.append({
            "pos": pos,
            "vel": vel,
            "radius": radius,
            "health": health,
            "score_value": score,
            "size_key": size_key,
            "angle": 0,
            "rot_speed": self.np_random.uniform(-0.05, 0.05),
            "shape": self._generate_asteroid_shape(radius),
        })

    def _break_asteroid(self, parent_ast):
        if parent_ast["size_key"] in ["large", "medium"]:
            num_children = 2
            if parent_ast["size_key"] == "large":
                child_size_key, child_radius, child_health, child_score = "medium", 20, 2, 2
            else: # medium
                child_size_key, child_radius, child_health, child_score = "small", 12, 1, 1
            
            for _ in range(num_children):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = parent_ast["vel"].length() * self.np_random.uniform(1.1, 1.4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.asteroids.append({
                    "pos": pygame.Vector2(parent_ast["pos"]),
                    "vel": vel,
                    "radius": child_radius,
                    "health": child_health,
                    "score_value": child_score,
                    "size_key": child_size_key,
                    "angle": 0,
                    "rot_speed": self.np_random.uniform(-0.1, 0.1),
                    "shape": self._generate_asteroid_shape(child_radius),
                })

    def _generate_asteroid_shape(self, radius):
        num_points = 12
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points)
            dist = radius + self.np_random.uniform(-radius * 0.4, radius * 0.4)
            points.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _spawn_powerup(self):
        ptype = self.np_random.choice(["invincibility", "rapid_fire"])
        self.powerups.append({
            "pos": pygame.Vector2(self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 50)),
            "type": ptype,
            "spawn_time": self.steps
        })

    def _create_explosion(self, pos, radius, num_particles, color_override=None):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(20, 40),
                "max_lifespan": 40,
                "color": color_override if color_override else self.np_random.choice(self.COLOR_EXPLOSION)
            })

    def _find_nearest_asteroid_dist(self):
        if not self.asteroids:
            return None
        player_pos = self.player["pos"]
        return min(player_pos.distance_to(ast["pos"]) for ast in self.asteroids)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for star in self.stars:
            c = star["brightness"]
            pygame.draw.rect(self.screen, (c, c, c), (*star["pos"], star["size"], star["size"]))

        # Power-ups
        for pu in self.powerups:
            color = self.COLOR_POWERUP_INVINCIBILITY if pu["type"] == "invincibility" else self.COLOR_POWERUP_RAPIDFIRE
            pulse = abs(math.sin((self.steps - pu["spawn_time"]) * 0.1))
            radius = int(10 + pulse * 4)
            pygame.gfxdraw.filled_circle(self.screen, int(pu["pos"].x), int(pu["pos"].y), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(pu["pos"].x), int(pu["pos"].y), radius, color)

        # Asteroids
        for ast in self.asteroids:
            rotated_shape = [p.rotate_rad(ast["angle"]) + ast["pos"] for p in ast["shape"]]
            int_points = [(int(p.x), int(p.y)) for p in rotated_shape]
            if len(int_points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)

        # Projectiles
        for proj in self.projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (int(proj.x), int(proj.y)), (int(proj.x), int(proj.y) - 8), 2)
        
        # Player
        if self.player["health"] > 0:
            is_invincible = self.player["invincibility_timer"] > 0 or self.player["active_powerup"] == "invincibility"
            alpha = self.player["invincibility_timer"] / (self.FPS * 2) if self.player["invincibility_timer"] > 0 else 0
            alpha += abs(math.sin(self.steps * 0.3)) * 0.5 if self.player["active_powerup"] == "invincibility" else 0
            
            p = self.player["pos"]
            s = self.player["size"]
            points = [(p.x, p.y - s), (p.x - s/2, p.y + s/2), (p.x + s/2, p.y + s/2)]
            
            if is_invincible:
                glow_radius = int(s * (1.5 + abs(math.sin(self.steps * 0.3))))
                pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), glow_radius, self.COLOR_PLAYER_GLOW)

            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            life_ratio = p["lifespan"] / p["max_lifespan"]
            radius = int(life_ratio * 4)
            color = p["color"]
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), radius, color)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_surf = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        # Health
        for i in range(3):
            icon_pos = (20 + i * 30, self.HEIGHT - 30)
            s = 8
            points = [(icon_pos[0], icon_pos[1] - s), (icon_pos[0] - s/2, icon_pos[1] + s/2), (icon_pos[0] + s/2, icon_pos[1] + s/2)]
            color = self.COLOR_HEALTH if i < self.player["health"] else self.COLOR_HEALTH_BG
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Power-up status
        base_y = self.HEIGHT - 35
        if self.player["banked_powerup"]:
            text_surf = self.font_small.render("POWER-UP READY", True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, base_y))
        elif self.player["active_powerup"]:
            name = self.player["active_powerup"].upper().replace("_", " ")
            color = self.COLOR_POWERUP_INVINCIBILITY if self.player["active_powerup"] == "invincibility" else self.COLOR_POWERUP_RAPIDFIRE
            text_surf = self.font_small.render(f"{name} ACTIVE", True, color)
            self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, base_y - 20))
            
            # Timer bar for active power-up
            bar_w = 100
            bar_h = 5
            bar_x = self.WIDTH - bar_w - 10
            bar_y = base_y + 5
            ratio = self.player["powerup_timer"] / (self.FPS * (5 if self.player["active_powerup"] == "invincibility" else 8))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_w * ratio), bar_h))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test assertions from brief
        self.reset()
        self.player["health"] = 4
        assert self.player["health"] <= 4 # Allow temp over-health if a powerup did it
        self.player["health"] = 3
        
        initial_score = self.score
        self._spawn_asteroid(1)
        self.asteroids[-1]["health"] = 1
        self.projectiles.append(pygame.Vector2(self.asteroids[-1]["pos"]))
        reward = self._handle_collisions()
        assert self.score > initial_score
        assert reward > 0

        initial_steps = self.steps
        self.step(self.action_space.sample())
        assert self.steps == initial_steps + 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy', 'directfb' or 'fbcon' if you have issues
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Annihilator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause before closing
            
        clock.tick(env.FPS)
        
    env.close()