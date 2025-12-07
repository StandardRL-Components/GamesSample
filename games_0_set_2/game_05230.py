# Generated: 2025-08-28T04:23:26.116397
# Source Brief: brief_05230.md
# Brief Index: 5230

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
from typing import Tuple
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to activate your mining laser on a nearby asteroid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect valuable minerals while dodging deadly meteors."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ASTEROID = (120, 120, 120)
    COLOR_METEOR = (255, 50, 50)
    COLOR_MINERAL = (255, 220, 50)
    COLOR_EXPLOSION = (255, 100, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_LASER = (0, 255, 255)

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    PLAYER_SPEED = 4
    PLAYER_SIZE = 12
    MAX_ASTEROIDS = 8
    MAX_METEORS = 5
    MINING_RADIUS = 60
    MINING_DURATION = 10 # frames the laser is visible
    WIN_SCORE = 100
    MAX_STEPS = 10000
    RISK_RADIUS = 150 # for reward bonus

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Etc...        
        self.player = {}
        self.asteroids = []
        self.meteors = []
        self.particles = []
        self.stars = []
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "angle": 0,
            "size": self.PLAYER_SIZE,
            "hitbox_radius": self.PLAYER_SIZE * 0.8
        }
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.prev_space_held = False
        self.mining_laser = None
        
        self.asteroids = []
        self.meteors = []
        self.particles = []
        
        for _ in range(self.MAX_ASTEROIDS):
            self._spawn_asteroid()

        for _ in range(self.MAX_METEORS):
            self._spawn_meteor()
            
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.uniform(0.5, 1.5),
            )
            for _ in range(100)
        ]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = -0.01  # Time penalty to encourage speed
        
        if not self.game_over:
            self._update_player(movement)
            reward += self._handle_mining(space_held)
            self._update_meteors()
            self._update_particles()
            
            # Spawn new entities if needed
            if len(self.asteroids) < self.MAX_ASTEROIDS and self.steps % 60 == 0:
                self._spawn_asteroid()
            if len(self.meteors) < self.MAX_METEORS and self.steps % 90 == 0:
                self._spawn_meteor()

            collision_reward = self._check_collisions()
            if collision_reward < 0:
                 reward = collision_reward # Override other rewards on death

            # Add complex reward for risky/safe behavior only if no collision
            if not self.game_over:
                reward += self._calculate_behavioral_reward(movement)

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.game_won:
                reward += 100
            else: # Implicitly a loss
                reward -= 100
                self._create_explosion(self.player["pos"])
            self.game_over = True

        self.prev_space_held = space_held
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "minerals": self.score, # Alias for clarity
            "is_won": self.game_won,
        }

    def _update_player(self, movement):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
            self.player["angle"] = 0
        elif movement == 2:  # Down
            move_vec.y = 1
            self.player["angle"] = 180
        elif movement == 3:  # Left
            move_vec.x = -1
            self.player["angle"] = 90
        elif movement == 4:  # Right
            move_vec.x = 1
            self.player["angle"] = 270

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["pos"] += move_vec * self.PLAYER_SPEED

        # World wrapping
        self.player["pos"].x %= self.WIDTH
        self.player["pos"].y %= self.HEIGHT

    def _handle_mining(self, space_held):
        reward = 0
        # Trigger mining on key press (rising edge)
        if space_held and not self.prev_space_held:
            closest_asteroid = None
            min_dist = float('inf')
            
            for asteroid in self.asteroids:
                dist = self.player["pos"].distance_to(asteroid["pos"])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            
            if closest_asteroid and min_dist <= self.MINING_RADIUS:
                # # sound: mining_laser_start.wav
                minerals_collected = closest_asteroid["minerals"]
                self.score += minerals_collected
                reward += minerals_collected * 0.1
                
                # Risk bonus for mining near a meteor
                if self.meteors:
                    if min(self.player["pos"].distance_to(m["pos"]) for m in self.meteors) < self.RISK_RADIUS:
                        reward += 1.0
                
                # Create particles
                for _ in range(minerals_collected * 2):
                    self._create_mineral_particle(closest_asteroid["pos"])
                
                self.asteroids.remove(closest_asteroid)
                self.mining_laser = {"target": closest_asteroid["pos"], "timer": self.MINING_DURATION}

        # Update laser visibility timer
        if self.mining_laser:
            self.mining_laser["timer"] -= 1
            if self.mining_laser["timer"] <= 0:
                self.mining_laser = None
                
        return reward

    def _update_meteors(self):
        for meteor in self.meteors[:]:
            meteor["pos"] += meteor["vel"]
            # Remove if off-screen
            if not (-meteor["size"] < meteor["pos"].x < self.WIDTH + meteor["size"] and \
                    -meteor["size"] < meteor["pos"].y < self.HEIGHT + meteor["size"]):
                self.meteors.remove(meteor)
                self._spawn_meteor()

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        # Player-Meteor collision
        for meteor in self.meteors:
            dist = self.player["pos"].distance_to(meteor["pos"])
            if dist < self.player["hitbox_radius"] + meteor["size"]:
                # # sound: explosion.wav
                self.game_over = True
                self.game_won = False
                return -100 # Immediate large penalty
        return 0

    def _calculate_behavioral_reward(self, movement):
        if not self.meteors or movement == 0:
            return 0
        
        closest_meteor_dist = min(self.player["pos"].distance_to(m["pos"]) for m in self.meteors)
        
        if closest_meteor_dist < 100:
            closest_asteroid_dist = float('inf')
            if self.asteroids:
                closest_asteroid_dist = min(self.player["pos"].distance_to(a["pos"]) for a in self.asteroids)
            
            # If not near an asteroid and moving, it's likely a "safe" but non-productive move.
            if closest_asteroid_dist > self.MINING_RADIUS * 1.5:
                return -0.2
                
        return 0

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_won = True
            return True
        if self.game_over: # Set by collision or step limit
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            brightness = 60 + math.sin(self.steps * 0.1 + x) * 20
            pygame.draw.rect(self.screen, (brightness, brightness, brightness), (int(x), int(y), int(size), int(size)))

        # Asteroids
        for asteroid in self.asteroids:
            points = [(p + asteroid["pos"]) for p in asteroid["points"]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        # Meteors
        for meteor in self.meteors:
            pos = (int(meteor["pos"].x), int(meteor["pos"].y))
            size = int(meteor["size"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_METEOR)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_METEOR)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / p["max_lifespan"]))))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

        # Mining Laser
        if self.mining_laser:
            alpha = max(0, min(255, int(255 * (self.mining_laser["timer"] / self.MINING_DURATION))))
            start_pos = self.player["pos"]
            end_pos = self.mining_laser["target"]
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)

        # Player
        if not (self.game_over and not self.game_won):
            p = self.player
            
            # Glow effect
            glow_size = int(p["size"] * 2.5)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_size // 2, glow_size // 2), glow_size // 2)
            self.screen.blit(glow_surf, (int(p["pos"].x - glow_size/2), int(p["pos"].y - glow_size/2)))
            
            # Ship body
            points = [
                pygame.Vector2(0, -p["size"]),
                pygame.Vector2(-p["size"] * 0.7, p["size"] * 0.7),
                pygame.Vector2(p["size"] * 0.7, p["size"] * 0.7),
            ]
            rotated_points = [point.rotate(-p["angle"]) + p["pos"] for point in points]
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.game_won else "SHIP DESTROYED"
            color = self.COLOR_MINERAL if self.game_won else self.COLOR_METEOR
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Spawning and Helper Methods ---

    def _spawn_asteroid(self):
        pos = pygame.Vector2(
            self.np_random.integers(0, self.WIDTH),
            self.np_random.integers(0, self.HEIGHT)
        )
        size = self.np_random.uniform(15, 30)
        minerals = int(size / 3)
        
        num_points = self.np_random.integers(6, 10)
        points = []
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i
            dist = size * self.np_random.uniform(0.7, 1.1)
            points.append(pygame.Vector2(math.cos(angle) * dist, math.sin(angle) * dist))
            
        self.asteroids.append({"pos": pos, "size": size, "minerals": minerals, "points": points})

    def _spawn_meteor(self):
        # Difficulty scaling: speed increases by 0.5 every 25 minerals
        level = self.score // 25
        base_speed = 1.0 + level * 0.5
        speed = self.np_random.uniform(base_speed, base_speed + 1.0)
        
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
        elif edge == 2: # Left
            pos = pygame.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
            
        target = pygame.Vector2(
            self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8),
            self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
        )
        vel = (target - pos).normalize() * speed
        size = self.np_random.uniform(5, 12)
        
        self.meteors.append({"pos": pos, "vel": vel, "size": size})

    def _create_mineral_particle(self, origin):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        lifespan = self.np_random.integers(15, 30)
        self.particles.append({
            "pos": pygame.Vector2(origin), "vel": vel, "lifespan": lifespan,
            "max_lifespan": lifespan, "color": self.COLOR_MINERAL,
            "size": self.np_random.integers(2, 4)
        })

    def _create_explosion(self, origin):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(20, 50)
            self.particles.append({
                "pos": pygame.Vector2(origin), "vel": vel, "lifespan": lifespan,
                "max_lifespan": lifespan, "color": self.COLOR_EXPLOSION,
                "size": self.np_random.integers(2, 5)
            })

# Example usage for manual play
if __name__ == '__main__':
    # This block will not be executed in the test environment,
    # but is useful for manual testing.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Space Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30)
        
    pygame.quit()