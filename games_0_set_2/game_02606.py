
# Generated: 2025-08-27T20:53:04.558165
# Source Brief: brief_02606.md
# Brief Index: 2606

        
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
        "Controls: ←→ to move. ↑ for a small jump, ↑ + Shift for a large jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated, side-scrolling corridor filled with "
        "pits and obstacles, jumping to survive in a chilling horror setting."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLATFORM = (40, 40, 55)
    COLOR_PLATFORM_EDGE = (60, 60, 80)
    COLOR_PLAYER = (230, 230, 240)
    COLOR_PLAYER_GLOW = (180, 180, 220, 30)
    COLOR_DANGER_GLOW = (200, 0, 0, 15)
    COLOR_OBSTACLE = (100, 30, 30)
    COLOR_OBSTACLE_EDGE = (140, 50, 50)
    COLOR_GOAL = (50, 200, 50)
    COLOR_GOAL_GLOW = (80, 255, 80, 40)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEART = (200, 20, 20)

    # Physics
    PLAYER_SPEED = 5
    GRAVITY = 0.8
    JUMP_STRENGTH_SMALL = -11
    JUMP_STRENGTH_LARGE = -15
    MAX_FALL_SPEED = 15

    # Level
    LEVEL_LENGTH_X = 6000
    GROUND_Y = 350
    PLATFORM_HEIGHT = 50

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.time_left = None
        self.platforms = None
        self.obstacles = None
        self.end_goal_rect = None
        self.camera_x = None
        self.particles = None
        self.last_platform_idx = None
        self.screen_shake = 0
        
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback to default if no seed is provided
            if self.np_random is None:
                 self.np_random = np.random.default_rng()


        self.player_pos = pygame.Vector2(150, self.GROUND_Y - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 20, 40)
        self.is_grounded = False
        self.last_platform_idx = 0

        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.time_left = 90 * self.FPS  # 90 seconds

        self.camera_x = 0
        self.particles = []
        self.screen_shake = 0

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        reward = 0.1  # Reward for surviving a step

        self._handle_input(action)
        reward += self._update_player()
        self._update_camera()
        self._update_particles()

        terminated = self._check_termination()
        if terminated:
            if self.player_pos.x >= self.end_goal_rect.x:
                reward += 100 # Reached the end
                self.score += 100
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
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        shift_held = action[2] == 1

        # Horizontal Movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # Jumping
        if movement == 1 and self.is_grounded:  # Up
            jump_strength = self.JUMP_STRENGTH_LARGE if shift_held else self.JUMP_STRENGTH_SMALL
            self.player_vel.y = jump_strength
            self.is_grounded = False
            # sfx: jump_sound
            self._create_particles(self.player_rect.midbottom, 15, (100,100,120), 'jump')

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Move player
        self.player_pos += self.player_vel

        # Prevent moving before start line
        self.player_pos.x = max(self.player_pos.x, self.player_rect.width / 2)
        
        # Update player rect
        self.player_rect.midbottom = self.player_pos

        # Check for obstacle collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs):
                # A simple bounce-back for now
                if self.player_vel.x > 0 and self.player_rect.right > obs.left:
                    self.player_rect.right = obs.left
                elif self.player_vel.x < 0 and self.player_rect.left < obs.right:
                    self.player_rect.left = obs.right
                if self.player_vel.y > 0 and self.player_rect.bottom > obs.top:
                     self.player_rect.bottom = obs.top
                     self.player_vel.y = 0
                self.player_pos.midbottom = self.player_rect.midbottom


        # Check for ground collision
        self.is_grounded = False
        landed_reward = 0
        
        for i, plat in enumerate(self.platforms):
            if (self.player_rect.colliderect(plat) and self.player_vel.y >= 0
                    and self.player_rect.bottom <= plat.top + self.player_vel.y + 1):
                self.player_pos.y = plat.top
                self.player_rect.bottom = plat.top
                self.player_vel.y = 0
                if not self.is_grounded: # First frame of landing
                    # sfx: land_sound
                    self._create_particles(self.player_rect.midbottom, 10, (100,100,120), 'land')
                    self.screen_shake = 5
                    
                    # Check for "challenging jump" reward
                    if self.last_platform_idx is not None and self.last_platform_idx != i:
                        last_plat = self.platforms[self.last_platform_idx]
                        gap = plat.left - last_plat.right
                        if gap > abs(self.JUMP_STRENGTH_SMALL) * self.PLAYER_SPEED * 0.8: # 80% of small jump range
                            landed_reward = 10

                self.is_grounded = True
                self.last_platform_idx = i
                break

        # Check for falling into a pit
        if self.player_pos.y > self.SCREEN_HEIGHT + self.player_rect.height:
            self.lives -= 1
            self.screen_shake = 15
            # sfx: fall_scream
            if self.lives > 0:
                # Respawn at the start of the last platform
                last_plat = self.platforms[self.last_platform_idx]
                self.player_pos = pygame.Vector2(last_plat.left + 50, last_plat.top - 50)
                self.player_vel = pygame.Vector2(0, 0)
            return -5 # Penalty for falling
        
        return landed_reward

    def _update_camera(self):
        # Camera follows player with a slight lag
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_x = self.camera_x * 0.95 + target_camera_x * 0.05
        # Clamp camera
        self.camera_x = max(0, self.camera_x)
        self.camera_x = min(self.LEVEL_LENGTH_X - self.SCREEN_WIDTH, self.camera_x)
        
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _check_termination(self):
        return (self.lives <= 0 or
                self.player_pos.x >= self.end_goal_rect.x or
                self.time_left <= 0 or
                self.steps >= 1000)

    def _generate_level(self):
        self.platforms = []
        self.obstacles = []
        
        # Start platform
        x = 0
        start_plat_width = 400
        self.platforms.append(pygame.Rect(x, self.GROUND_Y, start_plat_width, self.PLATFORM_HEIGHT))
        x += start_plat_width

        # Procedural generation loop
        while x < self.LEVEL_LENGTH_X - 500:
            # Difficulty scaling based on progress
            progress_ratio = x / self.LEVEL_LENGTH_X
            
            min_pit = 40 + 80 * progress_ratio
            max_pit = 100 + 150 * progress_ratio
            pit_width = self.np_random.uniform(min_pit, max_pit)
            x += pit_width

            min_plat = 250 - 150 * progress_ratio
            max_plat = 500 - 300 * progress_ratio
            platform_width = self.np_random.uniform(min_plat, max_plat)
            
            new_platform = pygame.Rect(x, self.GROUND_Y, platform_width, self.PLATFORM_HEIGHT)
            self.platforms.append(new_platform)
            
            # Add obstacles based on difficulty
            obstacle_prob = 0.1 + 0.5 * progress_ratio
            if self.np_random.random() < obstacle_prob:
                obs_w = self.np_random.integers(30, 60)
                obs_h = self.np_random.integers(40, 80)
                obs_x = new_platform.left + self.np_random.integers(0, int(new_platform.width - obs_w))
                obs_y = new_platform.top - obs_h
                self.obstacles.append(pygame.Rect(obs_x, obs_y, obs_w, obs_h))

            x += platform_width

        # End goal
        self.end_goal_rect = pygame.Rect(x, self.GROUND_Y - 50, 100, self.PLATFORM_HEIGHT + 50)
        self.platforms.append(pygame.Rect(x, self.GROUND_Y, 500, self.PLATFORM_HEIGHT)) # Final safe platform

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            render_offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake)
            render_offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake)
            
        camera_offset = pygame.Vector2(self.camera_x - render_offset_x, -render_offset_y)

        self._render_background(camera_offset)
        self._render_level(camera_offset)
        self._render_particles(camera_offset)
        self._render_player(camera_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, cam_off):
        # Parallax effect: slower scrolling background elements
        for i in range(0, self.LEVEL_LENGTH_X, 300):
            # "Eerie portraits" - abstract shapes
            p_x = i - cam_off.x * 0.5
            if -100 < p_x < self.SCREEN_WIDTH + 100:
                color = (20,20,35)
                pygame.draw.rect(self.screen, color, (p_x, 100, 40, 60))
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (p_x, 100, 40, 60), 1)

    def _render_level(self, cam_off):
        # Render platforms
        for plat in self.platforms:
            screen_rect = plat.move(-cam_off.x, -cam_off.y)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_EDGE, screen_rect, 2)
        
        # Render obstacles
        for obs in self.obstacles:
            screen_rect = obs.move(-cam_off.x, -cam_off.y)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_EDGE, screen_rect, 2)

        # Render pit glows
        for i in range(len(self.platforms) - 1):
            p1 = self.platforms[i]
            p2 = self.platforms[i+1]
            gap_rect = pygame.Rect(p1.right, self.GROUND_Y, p2.left - p1.right, self.SCREEN_HEIGHT - self.GROUND_Y)
            screen_rect = gap_rect.move(-cam_off.x, -cam_off.y)
            if self.screen.get_rect().colliderect(screen_rect):
                # Using gfxdraw for alpha blending
                radius = int(min(screen_rect.width / 2, 50))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(screen_rect.centerx), int(screen_rect.top) + 30, radius, self.COLOR_DANGER_GLOW)


        # Render Goal
        screen_rect = self.end_goal_rect.move(-cam_off.x, -cam_off.y)
        if self.screen.get_rect().colliderect(screen_rect):
            pygame.draw.rect(self.screen, self.COLOR_GOAL, screen_rect)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_rect.centerx), int(screen_rect.centery), 50, self.COLOR_GOAL_GLOW)


    def _render_player(self, cam_off):
        screen_rect = self.player_rect.move(-cam_off.x, -cam_off.y)
        
        # Glow effect
        glow_radius = int(screen_rect.height * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, screen_rect.centerx, screen_rect.centery, glow_radius, self.COLOR_PLAYER_GLOW)

        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_rect, border_radius=3)

    def _render_particles(self, cam_off):
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'].x - cam_off.x), int(p['pos'].y - cam_off.y))
                pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))
    
    def _render_ui(self):
        # Lives
        for i in range(self.lives):
            pygame.draw.circle(self.screen, self.COLOR_HEART, (30 + i * 30, 30), 10)
            pygame.draw.circle(self.screen, (0,0,0), (30 + i * 30, 30), 10, 1)

        # Timer
        time_text = f"TIME: {self.time_left // self.FPS:02d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 20, 20))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, self.SCREEN_HEIGHT - 30))

        if self.game_over:
            msg = "YOU REACHED THE END!" if self.player_pos.x >= self.end_goal_rect.x else "GAME OVER"
            end_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_surf, (self.SCREEN_WIDTH // 2 - end_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_surf.get_height() // 2))

    def _create_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(0.5, 2))
            elif p_type == 'land':
                 vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-1, -0.5))
            else:
                 vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_x": self.player_pos.x,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset and initial observation
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Corridor Horror")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose the observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(env.FPS)
        
    env.close()