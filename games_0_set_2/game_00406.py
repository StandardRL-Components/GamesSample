
# Generated: 2025-08-27T13:33:26.868902
# Source Brief: brief_00406.md
# Brief Index: 406

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to move your ship. "
        "Collect power-ups and press space to activate your held power-up."
    )

    game_description = (
        "Pilot an alien ship through a treacherous asteroid field. "
        "Dodge obstacles, collect power-ups, and survive long enough to escape."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.screen_width = 640
        self.screen_height = 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.MAX_STEPS = 5000
        self.WIN_STEPS = 4000
        self.PLAYER_ACCEL = 1.2
        self.PLAYER_FRICTION = 0.9
        self.PLAYER_MAX_SPEED = 10
        self.PLAYER_RADIUS = 12

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ASTEROID = (255, 80, 80)
        self.COLOR_TEXT = (255, 255, 255)
        self.POWERUP_COLORS = {
            "shield": (0, 180, 255),
            "speed": (255, 255, 0),
            "score": (200, 0, 255)
        }
        
        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables (will be initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_vel = None
        self.asteroids = None
        self.powerups = None
        self.particles = None
        self.stars = None
        self.base_asteroid_speed = None
        self.held_powerup = None
        self.active_powerups = None
        self.prev_space_held = None
        self.rng = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = random.Random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.screen_width * 0.15, self.screen_height / 2]
        self.player_vel = 0
        
        self.asteroids = []
        self.powerups = []
        self.particles = []
        
        self.base_asteroid_speed = 2.0
        
        self.held_powerup = None
        self.active_powerups = {}
        self.prev_space_held = False

        self.stars = [
            {
                "pos": [self.rng.randint(0, self.screen_width), self.rng.randint(0, self.screen_height)],
                "speed": 1 + self.rng.random() * 2,
                "size": self.rng.randint(1, 2)
            } for _ in range(150)
        ]

        for _ in range(5):
            self._spawn_asteroid(initial=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input & Calculate Movement Reward ---
        movement_intent = self._handle_input(action)
        reward += self._calculate_movement_reward(movement_intent)
        
        # --- 2. Update Game Logic ---
        self._update_player()
        self._update_stars()
        self._update_particles()
        self._update_active_powerups()
        
        reward += self._update_asteroids()
        reward += self._update_powerups()

        self._spawn_entities()
        
        # --- 3. Collision Detection ---
        collision_reward, terminated_by_collision = self._check_collisions()
        reward += collision_reward
        
        # --- 4. Step-based Rewards & Difficulty Scaling ---
        self.steps += 1
        reward += 0.1  # Survival reward
        if self.player_pos[1] > self.screen_height / 2:
            reward -= 0.2  # Penalty for being in riskier bottom half

        if self.steps % 500 == 0:
            self.base_asteroid_speed += 0.05

        # --- 5. Termination Conditions ---
        terminated = terminated_by_collision or self.steps >= self.MAX_STEPS
        
        if not terminated and self.steps >= self.WIN_STEPS:
            terminated = True
            reward += 100  # Win reward
            self._create_win_effect()
            self.game_over = True
        
        if terminated:
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        movement_intent = 0

        # Movement
        if movement == 1:  # Up
            self.player_vel -= self.PLAYER_ACCEL
            movement_intent = -1
        elif movement == 2:  # Down
            self.player_vel += self.PLAYER_ACCEL
            movement_intent = 1
        
        # Power-up Activation
        space_held = space_action == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if space_pressed and self.held_powerup:
            # SFX: Powerup_Activate.wav
            self._activate_powerup(self.held_powerup)
            self.held_powerup = None
        
        return movement_intent

    def _calculate_movement_reward(self, movement_intent):
        if movement_intent == 0 or not self.powerups:
            return 0
        
        # Find nearest power-up
        player_y = self.player_pos[1]
        closest_powerup = min(self.powerups, key=lambda p: abs(p['pos'][0] - self.player_pos[0]))
        powerup_y = closest_powerup['pos'][1]
        
        # Reward for moving towards it
        if (movement_intent < 0 and powerup_y < player_y) or \
           (movement_intent > 0 and powerup_y > player_y):
            return 0.5
        return 0

    def _update_player(self):
        speed_multiplier = 2.0 if "speed" in self.active_powerups else 1.0
        self.player_vel = np.clip(self.player_vel, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_pos[1] += self.player_vel * speed_multiplier
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.screen_height - self.PLAYER_RADIUS)

    def _update_stars(self):
        for star in self.stars:
            star["pos"][0] -= star["speed"]
            if star["pos"][0] < 0:
                star["pos"][0] = self.screen_width
                star["pos"][1] = self.rng.randint(0, self.screen_height)

    def _update_asteroids(self):
        near_miss_reward = 0
        for asteroid in self.asteroids[:]:
            asteroid["pos"][0] -= asteroid["speed"]
            asteroid["angle"] += asteroid["rot_speed"]
            
            # Near miss check
            if not asteroid.get("missed", False):
                dist = math.hypot(self.player_pos[0] - asteroid["pos"][0], self.player_pos[1] - asteroid["pos"][1])
                if dist < asteroid["radius"] + self.PLAYER_RADIUS + 30:
                    asteroid["missed"] = True
                    near_miss_reward += 1.0
                    self._create_particles(asteroid["pos"], self.COLOR_ASTEROID, 5, 1, 2)
                    # SFX: Near_Miss_Whoosh.wav
            
            if asteroid["pos"][0] < -asteroid["radius"]:
                self.asteroids.remove(asteroid)
        return near_miss_reward

    def _update_powerups(self):
        for powerup in self.powerups[:]:
            powerup["pos"][0] -= self.base_asteroid_speed * 0.8
            if powerup["pos"][0] < -powerup["radius"]:
                self.powerups.remove(powerup)
        return 0

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_active_powerups(self):
        for p_type in list(self.active_powerups.keys()):
            self.active_powerups[p_type] -= 1
            if self.active_powerups[p_type] <= 0:
                del self.active_powerups[p_type]
                # SFX: Powerup_Deactivate.wav

    def _spawn_entities(self):
        if self.rng.random() < 0.03: # Asteroid spawn rate
            self._spawn_asteroid()
        if self.rng.random() < 0.008 and len(self.powerups) < 2: # Power-up spawn rate
            self._spawn_powerup()

    def _spawn_asteroid(self, initial=False):
        x = self.screen_width + 50 if not initial else self.rng.randint(int(self.screen_width/2), self.screen_width)
        y = self.rng.randint(int(self.screen_height * 0.4), self.screen_height) # Spawn more in bottom half
        radius = self.rng.randint(15, 40)
        speed = self.base_asteroid_speed + self.rng.random() * 2
        self.asteroids.append({
            "pos": [x, y],
            "radius": radius,
            "speed": speed,
            "angle": 0,
            "rot_speed": self.rng.uniform(-0.05, 0.05),
            "shape": [(math.cos(a) * radius, math.sin(a) * radius) for a in sorted([self.rng.uniform(0, 2*math.pi) for _ in range(self.rng.randint(6, 9))])]
        })

    def _spawn_powerup(self):
        p_type = self.rng.choice(list(self.POWERUP_COLORS.keys()))
        self.powerups.append({
            "pos": [self.screen_width + 30, self.rng.randint(50, self.screen_height - 50)],
            "radius": 12,
            "type": p_type
        })

    def _check_collisions(self):
        # Player vs Asteroids
        for asteroid in self.asteroids:
            dist = math.hypot(self.player_pos[0] - asteroid["pos"][0], self.player_pos[1] - asteroid["pos"][1])
            if dist < self.PLAYER_RADIUS + asteroid["radius"]:
                if "shield" in self.active_powerups:
                    del self.active_powerups["shield"]
                    self.asteroids.remove(asteroid)
                    self._create_particles(self.player_pos, self.POWERUP_COLORS["shield"], 30, 2, 4)
                    # SFX: Shield_Block.wav
                    return 10, False # Reward for using shield
                else:
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, 2, 5)
                    # SFX: Player_Explosion.wav
                    return -100, True # Game over
        
        # Player vs Powerups
        for powerup in self.powerups[:]:
            dist = math.hypot(self.player_pos[0] - powerup["pos"][0], self.player_pos[1] - powerup["pos"][1])
            if dist < self.PLAYER_RADIUS + powerup["radius"]:
                if not self.held_powerup: # Can only hold one
                    self.held_powerup = powerup["type"]
                    self.powerups.remove(powerup)
                    self._create_particles(self.player_pos, self.POWERUP_COLORS[powerup["type"]], 20, 1, 3)
                    # SFX: Powerup_Collect.wav
                    return 5, False
        
        return 0, False

    def _create_particles(self, pos, color, count, min_speed, max_speed):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(min_speed, max_speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.randint(15, 30),
                "color": color
            })

    def _create_win_effect(self):
        for i in range(150):
            angle = (i / 150) * 2 * math.pi
            speed = self.rng.uniform(4, 8)
            self.particles.append({
                "pos": list(self.player_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.randint(40, 80),
                "color": self.rng.choice([self.COLOR_PLAYER, (255,255,0), (255,255,255)])
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_stars()
        self._render_particles()
        self._render_powerups()
        self._render_asteroids()
        if not self.game_over or self.steps >= self.WIN_STEPS:
             self._render_player()
        self._render_escape_marker()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), (int(star["pos"][0]), int(star["pos"][1])), star["size"])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            if alpha > 0:
                size = int(p["life"] / 5)
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p["color"], alpha), (size, size), size)
                self.screen.blit(s, (int(p["pos"][0]) - size, int(p["pos"][1]) - size))

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        radius = self.PLAYER_RADIUS
        
        # Glow
        for i in range(4):
            alpha = 80 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i * 3, (*self.COLOR_PLAYER, alpha))
        
        # Main ship body
        p1 = (pos[0] + radius, pos[1])
        p2 = (pos[0] - radius / 2, pos[1] - radius * 0.8)
        p3 = (pos[0] - radius / 2, pos[1] + radius * 0.8)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        
        # Shield effect
        if "shield" in self.active_powerups:
            shield_alpha = 100 + (self.active_powerups["shield"] % 10) * 10
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 5, (*self.POWERUP_COLORS["shield"], shield_alpha))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 6, (*self.POWERUP_COLORS["shield"], shield_alpha // 2))


    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(p[0] * 1.2, p[1] * 1.2) for p in asteroid["shape"]]
            rotated_glow = self._rotate_points(points, asteroid["angle"])
            translated_glow = [(p[0] + asteroid["pos"][0], p[1] + asteroid["pos"][1]) for p in rotated_glow]
            if len(translated_glow) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, translated_glow, (*self.COLOR_ASTEROID, 40))

            points = asteroid["shape"]
            rotated_points = self._rotate_points(points, asteroid["angle"])
            translated_points = [(p[0] + asteroid["pos"][0], p[1] + asteroid["pos"][1]) for p in rotated_points]
            if len(translated_points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, translated_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, translated_points, self.COLOR_ASTEROID)

    def _rotate_points(self, points, angle):
        return [(p[0] * math.cos(angle) - p[1] * math.sin(angle), 
                 p[0] * math.sin(angle) + p[1] * math.cos(angle)) for p in points]

    def _render_powerups(self):
        for powerup in self.powerups:
            pos = (int(powerup["pos"][0]), int(powerup["pos"][1]))
            radius = powerup["radius"]
            color = self.POWERUP_COLORS[powerup["type"]]
            
            # Glow
            glow_alpha = 100 + math.sin(self.steps * 0.1) * 50
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.8), (*color, glow_alpha / 2))
            
            # Icon
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_escape_marker(self):
        progress = min(1.0, self.steps / self.WIN_STEPS)
        y_pos = self.screen_height * (1.0 - progress)
        pygame.draw.line(self.screen, (255, 255, 0, 100), (self.screen_width - 10, self.screen_height), (self.screen_width - 10, y_pos), 3)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Held Power-up
        y_offset = 50
        if self.held_powerup:
            text = self.font_small.render("HELD:", True, self.COLOR_TEXT)
            self.screen.blit(text, (10, y_offset))
            color = self.POWERUP_COLORS[self.held_powerup]
            pygame.draw.rect(self.screen, color, (80, y_offset, 20, 20))
            y_offset += 30

        # Active Power-ups
        for p_type, time_left in self.active_powerups.items():
            color = self.POWERUP_COLORS[p_type]
            pygame.draw.rect(self.screen, color, (10, y_offset, 20, 20))
            
            time_bar_width = 100
            bar_width = int(time_bar_width * (time_left / (180 if p_type != 'shield' else 1))) # Shield is 1 hit, not timed
            pygame.draw.rect(self.screen, color, (40, y_offset + 5, bar_width, 10))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (40, y_offset + 5, time_bar_width, 10), 1)
            y_offset += 30

    def _activate_powerup(self, p_type):
        if p_type == "shield":
            self.active_powerups["shield"] = 1 # Represents 1 hit
        elif p_type == "speed":
            self.active_powerups["speed"] = 180 # 6 seconds at 30fps
        elif p_type == "score":
            # Score multiplier is handled by rewarding on activation for simplicity
            self.score += 50
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "held_powerup": self.held_powerup,
            "active_powerups": list(self.active_powerups.keys()),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")