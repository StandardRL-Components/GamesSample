
# Generated: 2025-08-27T14:46:03.766379
# Source Brief: brief_00786.md
# Brief Index: 786

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ← to jump left, → to jump right. Land on platforms to score. "
        "Reach the final platform before time runs out."
    )

    game_description = (
        "A fast-paced geometric platformer. Hop across procedurally generated platforms, "
        "balancing speed and risk to reach the end of the level. Avoid red obstacles and "
        "try to finish before the timer runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000
    TOTAL_TIME_SECONDS = 120

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER_BASE = (255, 255, 255)
    COLOR_PLAYER_FAST = (255, 100, 100)
    COLOR_PLATFORM = (0, 200, 100)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_PARTICLE = (220, 220, 220)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Physics
    GRAVITY = 0.5
    JUMP_POWER_Y = -10
    JUMP_POWER_X = 6
    AIR_DRAG = 0.98

    # Game Mechanics
    PLAYER_SIZE = 20
    PLATFORM_HEIGHT = 15
    NUM_PLATFORMS_GOAL = 30
    OBSTACLE_UPDATE_INTERVAL = 50
    DIFFICULTY_INTERVAL = 200
    INITIAL_PLATFORM_GAP = 20
    MAX_PLATFORM_GAP = 80
    PLATFORM_WIDTH_RANGE = (80, 200)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_on_ground = False
        self.current_platform_index = 0
        self.platforms = []
        self.obstacles = []
        self.particles = deque()
        self.camera_pos = np.array([0.0, 0.0])
        self.base_platform_gap = self.INITIAL_PLATFORM_GAP
        
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TOTAL_TIME_SECONDS * self.FPS
        self.base_platform_gap = self.INITIAL_PLATFORM_GAP

        # Player state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_on_ground = True
        self.current_platform_index = 0

        # World state
        self.platforms.clear()
        self.obstacles.clear()
        self.particles.clear()
        
        # Generate initial platforms
        first_platform = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 100,
            self.SCREEN_HEIGHT * 0.75,
            200,
            self.PLATFORM_HEIGHT
        )
        self.platforms.append(first_platform)
        self.player_pos[1] = first_platform.top - self.PLAYER_SIZE
        
        self._generate_platforms(self.NUM_PLATFORMS_GOAL)
        
        # Camera
        self.camera_pos = self.player_pos.copy() - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 2 / 3])

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_player_physics()
        
        reward += self._handle_collisions()
        
        self._update_obstacles()
        reward += self._check_obstacle_collisions()

        self._update_particles()
        self._update_difficulty()
        self._generate_platforms()
        self._update_camera()

        # --- Update Timers and Steps ---
        self.steps += 1
        self.timer -= 1
        
        # Continuous reward
        reward += 0.1 if self.is_on_ground else -0.1

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and self.current_platform_index >= self.NUM_PLATFORMS_GOAL - 1:
            reward += 100  # Goal reached reward

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if self.is_on_ground:
            if movement == 3:  # Left
                self.player_vel = np.array([-self.JUMP_POWER_X, self.JUMP_POWER_Y])
                self.is_on_ground = False
                # sfx: jump
            elif movement == 4:  # Right
                self.player_vel = np.array([self.JUMP_POWER_X, self.JUMP_POWER_Y])
                self.is_on_ground = False
                # sfx: jump

    def _update_player_physics(self):
        if not self.is_on_ground:
            self.player_vel[1] += self.GRAVITY
        
        self.player_vel[0] *= self.AIR_DRAG
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        landed_this_frame = False
        reward = 0

        if self.player_vel[1] > 0:  # Only check for landing if moving down
            for i, platform in enumerate(self.platforms):
                if player_rect.colliderect(platform):
                    # Check if player was above the platform in the previous frame
                    if self.player_pos[1] + self.PLAYER_SIZE - self.player_vel[1] <= platform.top + 1:
                        self.player_pos[1] = platform.top - self.PLAYER_SIZE
                        self.player_vel = np.array([0.0, 0.0])
                        if not self.is_on_ground:
                            reward += 1.0 # Landing bonus
                            self._create_landing_particles(15)
                            # sfx: land
                        self.is_on_ground = True
                        landed_this_frame = True
                        self.current_platform_index = i
                        break
        
        if not landed_this_frame:
            self.is_on_ground = False
        
        return reward

    def _check_obstacle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        reward = 0
        
        for i in range(len(self.obstacles) - 1, -1, -1):
            obstacle = self.obstacles[i]
            if player_rect.colliderect(obstacle['rect']):
                reward -= 5.0
                self.obstacles.pop(i)
                self._create_landing_particles(5, self.COLOR_OBSTACLE)
                # sfx: hit_obstacle
        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.base_platform_gap = min(self.MAX_PLATFORM_GAP, self.base_platform_gap + 1)

    def _update_obstacles(self):
        if self.steps > 0 and self.steps % self.OBSTACLE_UPDATE_INTERVAL == 0:
            self.obstacles.clear()
            # Add obstacles to a few upcoming platforms
            for i in range(self.current_platform_index + 1, min(len(self.platforms), self.current_platform_index + 5)):
                if self.np_random.random() < 0.4:  # 40% chance to spawn obstacle
                    platform = self.platforms[i]
                    obstacle_size = 15
                    pos_x = self.np_random.uniform(platform.left, platform.right - obstacle_size)
                    rect = pygame.Rect(pos_x, platform.top - obstacle_size, obstacle_size, obstacle_size)
                    self.obstacles.append({'rect': rect})

    def _generate_platforms(self, num_to_generate=1):
        while len(self.platforms) < self.NUM_PLATFORMS_GOAL and len(self.platforms) < self.current_platform_index + 15:
            last_platform = self.platforms[-1]
            gap = self.base_platform_gap + self.np_random.uniform(-10, 10)
            width = self.np_random.uniform(self.PLATFORM_WIDTH_RANGE[0], self.PLATFORM_WIDTH_RANGE[1])
            
            # Ensure y-pos is reachable
            dy = self.np_random.uniform(-50, 50)
            new_y = np.clip(last_platform.y + dy, self.SCREEN_HEIGHT * 0.4, self.SCREEN_HEIGHT * 0.9)

            new_x = last_platform.right + gap
            new_platform = pygame.Rect(new_x, new_y, width, self.PLATFORM_HEIGHT)
            self.platforms.append(new_platform)

    def _update_particles(self):
        for _ in range(len(self.particles)):
            particle = self.particles.popleft()
            particle['pos'] += particle['vel']
            particle['life'] -= 1
            if particle['life'] > 0:
                self.particles.append(particle)

    def _update_camera(self):
        target_pos = self.player_pos - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 2 / 3])
        self.camera_pos += (target_pos - self.camera_pos) * 0.1

    def _check_termination(self):
        if self.player_pos[1] > self.SCREEN_HEIGHT + self.PLAYER_SIZE:
            return True
        if self.timer <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if self.current_platform_index >= self.NUM_PLATFORMS_GOAL - 1:
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        cam_x, cam_y = int(self.camera_pos[0]), int(self.camera_pos[1])

        # Render platforms
        for platform in self.platforms:
            screen_rect = platform.move(-cam_x, -cam_y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BASE, screen_rect, 1)

        # Render obstacles
        for obstacle in self.obstacles:
            rect = obstacle['rect'].move(-cam_x, -cam_y)
            points = [
                (rect.centerx, rect.top),
                (rect.left, rect.bottom),
                (rect.right, rect.bottom)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0] - cam_x), int(p['pos'][1] - cam_y))
            radius = int(p['life'] / 5)
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, p['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

        # Render player
        speed = np.linalg.norm(self.player_vel)
        speed_ratio = min(1, speed / 15.0)
        player_color = (
            int(self.COLOR_PLAYER_BASE[0] * (1 - speed_ratio) + self.COLOR_PLAYER_FAST[0] * speed_ratio),
            int(self.COLOR_PLAYER_BASE[1] * (1 - speed_ratio) + self.COLOR_PLAYER_FAST[1] * speed_ratio),
            int(self.COLOR_PLAYER_BASE[2] * (1 - speed_ratio) + self.COLOR_PLAYER_FAST[2] * speed_ratio),
        )
        player_rect_screen = pygame.Rect(
            int(self.player_pos[0] - cam_x), int(self.player_pos[1] - cam_y),
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, player_color, player_rect_screen)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (20, 20), self.font_small)

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 150, 20), self.font_small)
        
        # Goal
        goal_text = f"PLATFORM: {self.current_platform_index + 1} / {self.NUM_PLATFORMS_GOAL}"
        self._draw_text(goal_text, (self.SCREEN_WIDTH / 2, 20), self.font_small, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer / self.FPS,
            "platform": self.current_platform_index + 1,
        }

    def _create_landing_particles(self, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            pos = self.player_pos + np.array([self.PLAYER_SIZE / 2, self.PLAYER_SIZE])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'color': color
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        move_action = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [move_action, space_action, shift_action]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(1000)

        clock.tick(GameEnv.FPS)
        
    env.close()