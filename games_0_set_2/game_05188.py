import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: → to run right, ← to run left, ↑ to jump. Avoid the red obstacles and reach the green finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Guide a robot through a procedurally generated obstacle course to reach the finish line as quickly as possible. You have 3 lives and 20 seconds!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    TIME_LIMIT = 20.0  # seconds
    MAX_STEPS = 1000
    LEVEL_WIDTH = 3200

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_GROUND = (60, 40, 40)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_EYE = (255, 255, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_FINISH = (50, 255, 50)
    COLOR_TEXT = (240, 240, 240)

    # Physics & Player
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    PLAYER_SPEED = 6
    PLAYER_WIDTH = 24
    PLAYER_HEIGHT = 36
    GROUND_Y = SCREEN_HEIGHT - 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 64)

        self.player = {}
        self.obstacles = []
        self.particles = []
        self.camera_x = 0

        # The reset call is deferred to the first use, but we can initialize attributes here.
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.game_timer = self.TIME_LIMIT
        self.last_player_x_for_reset = 100
        self.finish_line_rect = pygame.Rect(0,0,0,0)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.game_timer = self.TIME_LIMIT
        self.last_player_x_for_reset = 100

        self.player = {
            "pos": pygame.Vector2(100, self.GROUND_Y - self.PLAYER_HEIGHT),
            "vel": pygame.Vector2(0, 0),
            "on_ground": False,
            "rect": pygame.Rect(100, self.GROUND_Y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        }

        self.camera_x = 0
        self.particles = []
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.obstacles = []
        current_x = 400
        while current_x < self.LEVEL_WIDTH - self.SCREEN_WIDTH / 2:
            gap = self.np_random.integers(120, 250)
            current_x += gap

            obstacle_type = self.np_random.choice(['block_low', 'block_high', 'projectile_spawner'])

            if obstacle_type == 'block_low':
                height = self.np_random.integers(20, 40)
                self.obstacles.append({
                    "rect": pygame.Rect(current_x, self.GROUND_Y - height, 40, height),
                    "type": "block", "cleared": False
                })
            elif obstacle_type == 'block_high':
                height = self.np_random.integers(50, 80)
                self.obstacles.append({
                    "rect": pygame.Rect(current_x, self.GROUND_Y - height, 30, height),
                    "type": "block", "cleared": False
                })
            elif obstacle_type == 'projectile_spawner':
                self.obstacles.append({
                    "rect": pygame.Rect(current_x + 60, self.GROUND_Y - 50, 10, 10),
                    "type": "projectile", "cleared": False, "vel_x": -4,
                })

        self.finish_line_rect = pygame.Rect(self.LEVEL_WIDTH - 150, self.GROUND_Y - 100, 10, 100)

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self._handle_input(movement)

        prev_player_pos_x = self.player["pos"].x
        self._update_player_physics()

        difficulty_mult = 1.0 + 0.05 * (self.steps // 500)
        self._update_obstacles(difficulty_mult)
        self._update_particles()

        self._update_camera()

        # Reward for progress
        progress = self.player["pos"].x - prev_player_pos_x
        reward += progress * 0.01

        collision_reward = self._check_collisions()
        reward += collision_reward

        clear_reward = self._check_obstacle_clear()
        reward += clear_reward

        self.game_timer -= 1.0 / self.FPS
        self.steps += 1

        self._check_termination_conditions()

        if self.win:
            reward += 100
        elif self.game_over:
            reward -= 100

        self.score += reward
        terminated = self.game_over or self.win
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"].y = self.JUMP_STRENGTH
            self.player["on_ground"] = False
            # sfx: jump
            self._create_particles(self.player["pos"] + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT), 10, (200, 200, 200), 2)

        if movement == 3:
            self.player["vel"].x = -self.PLAYER_SPEED
        elif movement == 4:
            self.player["vel"].x = self.PLAYER_SPEED
        else:
            self.player["vel"].x = 0

    def _update_player_physics(self):
        # Apply gravity
        self.player["vel"].y += self.GRAVITY

        # Move player
        self.player["pos"] += self.player["vel"]

        # World bounds
        if self.player["pos"].x < 0:
            self.player["pos"].x = 0
        if self.player["pos"].x > self.LEVEL_WIDTH - self.PLAYER_WIDTH:
            self.player["pos"].x = self.LEVEL_WIDTH - self.PLAYER_WIDTH

        # Ground collision
        if self.player["pos"].y + self.PLAYER_HEIGHT > self.GROUND_Y:
            if not self.player["on_ground"]:
                 # sfx: land
                 self._create_particles(self.player["pos"] + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT), 5, (150, 120, 100), 1)
            self.player["pos"].y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player["vel"].y = 0
            self.player["on_ground"] = True
        else:
            self.player["on_ground"] = False

        self.player["rect"].topleft = self.player["pos"]

    def _update_obstacles(self, difficulty_mult):
        for obs in self.obstacles:
            if obs["type"] == "projectile":
                obs["rect"].x += obs["vel_x"] * difficulty_mult
                if obs["rect"].right < 0:
                    obs["rect"].left = self.LEVEL_WIDTH # Reset projectile

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_camera(self):
        target_camera_x = self.player["pos"].x - self.SCREEN_WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH - self.SCREEN_WIDTH))

    def _check_collisions(self):
        for obs in self.obstacles:
            if self.player["rect"].colliderect(obs["rect"]):
                self.lives -= 1
                # sfx: explosion
                self._create_particles(self.player["rect"].center, 30, self.COLOR_OBSTACLE, 5)
                self.player["pos"] = pygame.Vector2(self.last_player_x_for_reset, self.GROUND_Y - self.PLAYER_HEIGHT)
                self.player["vel"] = pygame.Vector2(0, 0)
                if self.lives <= 0:
                    self.game_over = True
                return -5 # Collision penalty
        return 0

    def _check_obstacle_clear(self):
        reward = 0
        player_center_x = self.player["rect"].centerx
        for obs in self.obstacles:
            if not obs["cleared"] and player_center_x > obs["rect"].right:
                obs["cleared"] = True
                reward += 1 # Reward for clearing an obstacle

        # Update checkpoint for respawn
        if self.player["pos"].x > self.last_player_x_for_reset + self.SCREEN_WIDTH * 0.8:
            self.last_player_x_for_reset = self.player["pos"].x

        return reward

    def _check_termination_conditions(self):
        if self.player["rect"].colliderect(self.finish_line_rect):
            self.win = True
            # sfx: win
        if self.game_timer <= 0:
            self.game_over = True
            # sfx: lose
        if self.lives <= 0:
            self.game_over = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_ground()
        self._render_obstacles()
        self._render_finish_line()
        self._render_player()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.LEVEL_WIDTH, 50):
            x = int(i - self.camera_x)
            if -50 < x < self.SCREEN_WIDTH + 50:
                 pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GROUND_Y), 1)
        for i in range(0, self.GROUND_Y, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_rect = obs["rect"].copy()
            screen_rect.x -= int(self.camera_x)
            if screen_rect.right > 0 and screen_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)

    def _render_finish_line(self):
        screen_rect = self.finish_line_rect.copy()
        screen_rect.x -= int(self.camera_x)
        if screen_rect.right > 0 and screen_rect.left < self.SCREEN_WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FINISH, screen_rect)
            for i in range(screen_rect.height // 10):
                if i % 2 == 0:
                    pygame.draw.rect(self.screen, self.COLOR_BG, (screen_rect.x, screen_rect.y + i * 10, 10, 10))

    def _render_player(self):
        screen_pos = self.player["pos"] - pygame.Vector2(self.camera_x, 0)

        # Body
        body_rect = pygame.Rect(int(screen_pos.x), int(screen_pos.y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)

        # Eye
        eye_pos_x = int(screen_pos.x + self.PLAYER_WIDTH * 0.6)
        eye_pos_y = int(screen_pos.y + self.PLAYER_HEIGHT * 0.3)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (eye_pos_x, eye_pos_y), 4)
        pygame.draw.circle(self.screen, (0,0,0), (eye_pos_x, eye_pos_y), 2)

        # Simple leg animation
        if not self.player["on_ground"]: # Jumping pose
            leg_y = screen_pos.y + self.PLAYER_HEIGHT * 0.9
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (screen_pos.x + 5, leg_y), (screen_pos.x + 5, leg_y + 5), 3)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (screen_pos.x + self.PLAYER_WIDTH - 5, leg_y), (screen_pos.x + self.PLAYER_WIDTH - 5, leg_y + 5), 3)
        elif self.player["vel"].x != 0: # Running
            leg_phase = (self.steps % 10) / 10.0
            leg1_len = math.sin(leg_phase * 2 * math.pi) * 5
            leg2_len = -math.sin(leg_phase * 2 * math.pi) * 5
            leg_y = screen_pos.y + self.PLAYER_HEIGHT - 2
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (screen_pos.x + 5, leg_y), (screen_pos.x + 5, leg_y + leg1_len + 5), 4)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (screen_pos.x + self.PLAYER_WIDTH - 5, leg_y), (screen_pos.x + self.PLAYER_WIDTH - 5, leg_y + leg2_len + 5), 4)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(self.camera_x, 0)
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Pygame.gfxdraw requires a tuple for color, not a list
                if isinstance(color_with_alpha, list):
                    color_with_alpha = tuple(color_with_alpha)
                try:
                    pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), size, color_with_alpha)
                    pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), size, color_with_alpha)
                except (TypeError, ValueError):
                    # Fallback if color format is incorrect
                    pass


    def _render_ui(self):
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))

        # Timer
        timer_text = self.font_ui.render(f"TIME: {self.game_timer:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Win/Lose Message
        if self.win:
            msg = self.font_msg.render("YOU WIN!", True, self.COLOR_FINISH)
            self.screen.blit(msg, (self.SCREEN_WIDTH/2 - msg.get_width()/2, self.SCREEN_HEIGHT/2 - msg.get_height()/2))
        elif self.game_over:
            msg = self.font_msg.render("GAME OVER", True, self.COLOR_OBSTACLE)
            self.screen.blit(msg, (self.SCREEN_WIDTH/2 - msg.get_width()/2, self.SCREEN_HEIGHT/2 - msg.get_height()/2))

    def _create_particles(self, pos, count, color, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, 1.0) * speed
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_mag
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'size': self.np_random.integers(2, 5)})

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_left": self.game_timer,
            "player_x": self.player["pos"].x,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass
    
    env = GameEnv()
    obs, info = env.reset()

    running = True
    total_reward = 0

    # To keep track of held keys
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
    }

    # Map Pygame keys to action indices
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Create a display for human playing
    pygame.display.set_caption("Robot Runner")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        movement_action = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
            elif event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Determine movement action based on held keys (right takes precedence over left)
        if keys_held[pygame.K_RIGHT]:
            movement_action = key_to_movement[pygame.K_RIGHT]
        elif keys_held[pygame.K_LEFT]:
            movement_action = key_to_movement[pygame.K_LEFT]

        # Up/Down can be simultaneous with Left/Right, but Up takes precedence
        if keys_held[pygame.K_UP]:
            movement_action = key_to_movement[pygame.K_UP]
        elif keys_held[pygame.K_DOWN]:
             movement_action = key_to_movement[pygame.K_DOWN]

        # Construct the MultiDiscrete action
        action = [
            movement_action,
            1 if keys_held[pygame.K_SPACE] else 0,
            1 if keys_held[pygame.K_LSHIFT] else 0
        ]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart or close the window.")
            # Wait for restart or quit
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_input = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_input = False

        env.clock.tick(env.FPS)

    env.close()