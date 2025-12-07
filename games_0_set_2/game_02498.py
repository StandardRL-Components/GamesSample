
# Generated: 2025-08-27T20:32:48.992941
# Source Brief: brief_02498.md
# Brief Index: 2498

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press Space to fire a ball in your last direction of movement. "
        "Survive for 30 seconds!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A frantic top-down dodgeball arena. Dodge incoming enemy balls and throw your own to "
        "destroy them. Survive for 30 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.WIN_TIME_SECONDS = 30
        self.WIN_STEPS = self.WIN_TIME_SECONDS * self.FPS
        self.MAX_STEPS = self.WIN_STEPS + self.FPS * 5  # Allow a few seconds buffer

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_ARENA = (200, 200, 220)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_PLAYER_BALL = (100, 255, 100)
        self.COLOR_PLAYER_BALL_GLOW = (200, 255, 200)
        self.COLOR_ENEMY_SLOW = (255, 80, 80)
        self.COLOR_ENEMY_MEDIUM = (255, 160, 80)
        self.COLOR_ENEMY_FAST = (255, 240, 80)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TIMER = (230, 230, 230)
        self.COLOR_HEART = (255, 50, 50)

        # Game parameters
        self.PLAYER_RADIUS = 12
        self.PLAYER_SPEED = 5
        self.PLAYER_BALL_RADIUS = 8
        self.PLAYER_BALL_SPEED = 12
        self.ENEMY_RADIUS = 15
        self.ENEMY_BASE_SPEED = 2.0
        self.ENEMY_SPAWN_RATE_INITIAL = 1.0 * self.FPS # Every 1 second
        self.ENEMY_SPAWN_RATE_FINAL = 0.3 * self.FPS # Every 0.3 seconds

        self.ARENA_CENTER = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        self.ARENA_RADIUS = min(self.WIDTH, self.HEIGHT) / 2 - 30

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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = None
        self.last_movement_dir = None
        self.player_lives = None
        self.player_ball = None
        self.last_space_held = None
        self.enemy_balls = None
        self.particles = None
        self.enemy_spawn_timer = None
        self.current_enemy_speed = None
        self.current_spawn_rate = None
        self.steps = 0
        self.score = 0
        self.game_over_message = ""

        # Call reset to set initial state
        # self.reset() # This is called by the Gym wrapper, no need to call it here.

        # Run validation check
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = self.ARENA_CENTER.copy().astype(float)
        self.last_movement_dir = np.array([0, -1.0])  # Default up
        self.player_lives = 3
        self.player_ball = None
        self.last_space_held = False

        self.enemy_balls = []
        self.particles = []
        self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE_INITIAL
        self.current_enemy_speed = self.ENEMY_BASE_SPEED
        self.current_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.game_over_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1
            reward += 0.01  # Small reward for surviving each frame

            # --- Handle Actions ---
            self._handle_input(action)

            # --- Update Game State ---
            hit_reward = self._update_player_ball()
            reward += hit_reward

            dodge_reward = self._update_enemy_balls()
            reward += dodge_reward

            self._spawn_enemies()
            self._update_difficulty()

        # --- Update Visuals (Particles) ---
        self._update_particles()

        # --- Check Termination ---
        if self.player_lives <= 0 and not self.game_over:
            terminated = True
            reward = -100.0  # Large penalty for losing
            self.game_over = True
            self.game_over_message = "GAME OVER"
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 100, 5)

        if self.steps >= self.WIN_STEPS and not self.game_over:
            terminated = True
            reward = 100.0  # Large reward for winning
            self.game_over = True
            self.win = True
            self.game_over_message = "YOU SURVIVED!"

        if self.steps >= self.MAX_STEPS:
            terminated = True # End episode if it runs too long

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0  # Up
        elif movement == 2: move_vec[1] = 1.0   # Down
        elif movement == 3: move_vec[0] = -1.0  # Left
        elif movement == 4: move_vec[0] = 1.0   # Right

        if np.linalg.norm(move_vec) > 0:
            self.last_movement_dir = move_vec / np.linalg.norm(move_vec)
            self.player_pos += self.last_movement_dir * self.PLAYER_SPEED

        # Clamp player to arena
        dist_from_center = np.linalg.norm(self.player_pos - self.ARENA_CENTER)
        if dist_from_center > self.ARENA_RADIUS - self.PLAYER_RADIUS:
            direction = (self.player_pos - self.ARENA_CENTER) / dist_from_center
            self.player_pos = self.ARENA_CENTER + direction * (self.ARENA_RADIUS - self.PLAYER_RADIUS)

        # Shooting
        if space_held and not self.last_space_held and self.player_ball is None:
            # sound: player_shoot.wav
            self.player_ball = {
                "pos": self.player_pos.copy(),
                "vel": self.last_movement_dir * self.PLAYER_BALL_SPEED,
                "hit_enemy": False,
            }
        self.last_space_held = space_held

    def _update_player_ball(self):
        if self.player_ball is None:
            return 0

        reward = 0
        self.player_ball["pos"] += self.player_ball["vel"]

        # Check for collision with enemy balls
        for enemy in self.enemy_balls[:]:
            if np.linalg.norm(self.player_ball["pos"] - enemy["pos"]) < self.PLAYER_BALL_RADIUS + self.ENEMY_RADIUS:
                # sound: enemy_hit.wav
                self.score += 10
                reward += 2.0  # Reward for hitting an enemy
                self._create_particles(enemy["pos"], enemy["color"], 30)
                self.enemy_balls.remove(enemy)
                self.player_ball["hit_enemy"] = True
                self.player_ball = None # Ball is destroyed on hit
                return reward

        # Check for collision with arena boundary
        if np.linalg.norm(self.player_ball["pos"] - self.ARENA_CENTER) > self.ARENA_RADIUS:
            if not self.player_ball["hit_enemy"]:
                reward += -0.2  # Penalty for missing
            self.player_ball = None

        return reward

    def _update_enemy_balls(self):
        reward = 0
        for ball in self.enemy_balls[:]:
            ball["pos"] += ball["vel"]

            # Check for collision with player
            if np.linalg.norm(ball["pos"] - self.player_pos) < self.ENEMY_RADIUS + self.PLAYER_RADIUS:
                # sound: player_hit.wav
                self.player_lives -= 1
                self.score -= 25
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, 3)
                self.enemy_balls.remove(ball)
                continue

            # Check for collision with arena boundary (successful dodge)
            if np.linalg.norm(ball["pos"] - self.ARENA_CENTER) > self.ARENA_RADIUS + self.ENEMY_RADIUS * 2:
                self.score += 5
                reward += 0.5  # Reward for a successful dodge
                self.enemy_balls.remove(ball)

        return reward

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            # sound: enemy_spawn.wav
            angle = self.np_random.uniform(0, 2 * math.pi)
            spawn_pos = self.ARENA_CENTER + np.array([math.cos(angle), math.sin(angle)]) * (self.ARENA_RADIUS + 40)

            target_radius = self.np_random.uniform(0, self.ARENA_RADIUS * 0.8)
            target_angle = self.np_random.uniform(0, 2 * math.pi)
            target_pos = self.ARENA_CENTER + np.array([math.cos(target_angle), math.sin(target_angle)]) * target_radius

            direction = (target_pos - spawn_pos)
            dist = np.linalg.norm(direction)
            if dist == 0: dist = 1
            velocity = direction / dist * self.current_enemy_speed

            # Determine color based on speed
            speed_ratio = (self.current_enemy_speed - self.ENEMY_BASE_SPEED) / (self.ENEMY_BASE_SPEED * 2) # Heuristic
            if speed_ratio < 0.33: color = self.COLOR_ENEMY_SLOW
            elif speed_ratio < 0.66: color = self.COLOR_ENEMY_MEDIUM
            else: color = self.COLOR_ENEMY_FAST

            self.enemy_balls.append({"pos": spawn_pos, "vel": velocity, "color": color})
            self.enemy_spawn_timer = self.current_spawn_rate

    def _update_difficulty(self):
        progress = min(self.steps / self.WIN_STEPS, 1.0)
        self.current_enemy_speed = self.ENEMY_BASE_SPEED + (self.ENEMY_BASE_SPEED * 2) * progress
        self.current_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL - (self.ENEMY_SPAWN_RATE_INITIAL - self.ENEMY_SPAWN_RATE_FINAL) * progress

    def _create_particles(self, pos, color, count, max_speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # Damping
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER[0]), int(self.ARENA_CENTER[1]), int(self.ARENA_RADIUS), self.COLOR_ARENA)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifetime"] / 20))
            color_with_alpha = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Enemy balls
        for ball in self.enemy_balls:
            self._draw_circle_with_glow(self.screen, ball["pos"], self.ENEMY_RADIUS, ball["color"], ball["color"])

        # Player ball
        if self.player_ball:
            self._draw_circle_with_glow(self.screen, self.player_ball["pos"], self.PLAYER_BALL_RADIUS, self.COLOR_PLAYER_BALL, self.COLOR_PLAYER_BALL_GLOW)

        # Player
        if self.player_lives > 0:
            self._draw_circle_with_glow(self.screen, self.player_pos, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        time_left = max(0, self.WIN_TIME_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 10))

        # Lives
        for i in range(self.player_lives):
            self._draw_heart(self.WIDTH - 30 - (i * 35), 25, 12, self.COLOR_HEART)

        # Game Over Message
        if self.game_over:
            color = self.COLOR_PLAYER_BALL if self.win else self.COLOR_ENEMY_SLOW
            end_text = self.font_large.render(self.game_over_message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _draw_circle_with_glow(self, surface, center, radius, color, glow_color):
        # Draw glow
        glow_radius = int(radius * 1.8)
        glow_alpha = 70
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*glow_color, glow_alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (int(center[0] - glow_radius), int(center[1] - glow_radius)))
        
        # Draw main circle
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)

    def _draw_heart(self, x, y, size, color):
        p1 = (x, y - size // 4)
        p2 = (x + size // 2, y - size // 2)
        p3 = (x + size, y - size // 4)
        p4 = (x, y + size // 2)
        p5 = (x - size, y - size // 4)
        p6 = (x - size // 2, y - size // 2)
        points = [p1, p2, p3, (x + size * 3/4, y + size / 4), (x, y + size), (x - size * 3/4, y + size/4), p6]
        
        # Simplified heart shape for gfxdraw
        p_list = [
            (x, y + size * 0.75),
            (x - size, y - size * 0.25),
            (x - size * 0.5, y - size * 0.75),
            (x, y - size * 0.25),
            (x + size * 0.5, y - size * 0.75),
            (x + size, y - size * 0.25),
        ]
        pygame.gfxdraw.aapolygon(self.screen, p_list, color)
        pygame.gfxdraw.filled_polygon(self.screen, p_list, color)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "time_left": max(0, self.WIN_TIME_SECONDS - (self.steps / self.FPS))
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

        print("✓ Implementation validated successfully")