
# Generated: 2025-08-28T03:52:38.258188
# Source Brief: brief_05068.md
# Brief Index: 5068

        
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

    user_guide = "Controls: ←→ to run, ↑ to jump."
    game_description = "Guide a robot through a procedurally generated obstacle course to reach the finish line as quickly as possible."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 50
    MAX_STEPS = 3000
    TOTAL_STAGES = 3

    # Colors
    COLOR_BG = (16, 16, 24)
    COLOR_GRID = (32, 32, 48)
    COLOR_PLAYER = (64, 160, 255)
    COLOR_PLAYER_GLOW = (64, 160, 255, 50)
    COLOR_OBSTACLE = (255, 64, 64)
    COLOR_FINISH = (64, 255, 64)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER = (255, 255, 0)
    COLOR_JUMP_PARTICLE = (200, 200, 255)
    COLOR_LAND_PARTICLE = (180, 180, 180)

    # Physics
    GRAVITY = 0.4
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.90
    PLAYER_JUMP_STRENGTH = -10.0
    PLAYER_MAX_SPEED = 5.0
    PLAYER_WIDTH, PLAYER_HEIGHT = 20, 32

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.is_grounded = False
        self.jump_buffer = 0
        
        self.obstacles = []
        self.finish_line_x = 0
        self.camera_x = 0.0
        self.particles = []
        self.obstacles_cleared = set()

        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False
        self.stage_time = 60.0

        self._generate_stage()
        self._reset_player()
        
        self.obstacles_cleared = set()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        reward = -0.01  # Time penalty
        
        # --- Handle Input ---
        if movement == 1 and self.is_grounded: # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump
            self._spawn_particles(15, self.player_rect.midbottom, self.COLOR_JUMP_PARTICLE, 'up')
            
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
            reward += 0.1 # Reward for moving right
        
        # --- Update Physics & Game State ---
        self._update_player_physics()
        
        # Check for cleared obstacles
        newly_cleared_reward = self._check_cleared_obstacles()
        reward += newly_cleared_reward

        self.stage_time -= 1.0 / self.FPS
        self.steps += 1
        
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos.y > self.HEIGHT + 50: # Fell into pit
            reward -= 100
            terminated = True
            # sfx: fall
        
        if self.stage_time <= 0: # Time ran out
            reward -= 100
            terminated = True
            # sfx: timeout
            
        if self.player_pos.x >= self.finish_line_x: # Reached finish line
            reward += 100
            if self.stage < self.TOTAL_STAGES:
                self.stage += 1
                self._generate_stage()
                self._reset_player()
                self.stage_time = 60.0
                # sfx: stage_complete
            else:
                terminated = True # Final victory
                # sfx: victory
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        # Collision with side of obstacle is checked in _update_player_physics
        if self.game_over:
            reward -= 10
            terminated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player_physics(self):
        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        # Apply gravity
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, 15) # Terminal velocity

        # --- Collision Detection ---
        # Move horizontally
        self.player_pos.x += self.player_vel.x
        self.player_rect.x = int(self.player_pos.x)
        for obstacle in self.obstacles:
            if self.player_rect.colliderect(obstacle):
                # Hit side of obstacle
                self.game_over = True # Terminate on side collision
                # sfx: crash
                return

        # Move vertically
        self.player_pos.y += self.player_vel.y
        self.player_rect.y = int(self.player_pos.y)
        self.is_grounded = False
        for obstacle in self.obstacles:
            if self.player_rect.colliderect(obstacle):
                if self.player_vel.y > 0: # Moving down
                    self.player_rect.bottom = obstacle.top
                    self.player_pos.y = self.player_rect.y
                    self.player_vel.y = 0
                    if not self.is_grounded:
                        # sfx: land
                        self._spawn_particles(10, self.player_rect.midbottom, self.COLOR_LAND_PARTICLE, 'side')
                    self.is_grounded = True
                elif self.player_vel.y < 0: # Moving up
                    self.player_rect.top = obstacle.bottom
                    self.player_pos.y = self.player_rect.y
                    self.player_vel.y = 0 # Bonk head

    def _generate_stage(self):
        self.obstacles.clear()
        
        # Difficulty parameters
        obstacle_density = 0.5 + 0.2 * (self.stage - 1)
        gap_reduction = 1.0 - 0.1 * (self.stage - 1)
        
        min_gap = 80 * gap_reduction
        max_gap = 160 * gap_reduction
        min_len = 100
        max_len = 300

        # Starting platform
        self.obstacles.append(pygame.Rect(0, self.HEIGHT - 40, 400, 100))
        
        current_x = 400.0
        current_y = self.HEIGHT - 40
        level_width = 6400

        while current_x < level_width:
            gap = self.np_random.uniform(min_gap, max_gap)
            current_x += gap
            
            if self.np_random.random() < obstacle_density:
                plat_len = self.np_random.uniform(min_len, max_len)
                
                # Ensure y is reachable
                max_y_diff = 100
                new_y = current_y + self.np_random.uniform(-max_y_diff, max_y_diff)
                new_y = np.clip(new_y, 150, self.HEIGHT - 40)
                
                self.obstacles.append(pygame.Rect(int(current_x), int(new_y), int(plat_len), 100))
                current_x += plat_len
                current_y = new_y
        
        self.finish_line_x = current_x + 200

    def _reset_player(self):
        self.player_pos = pygame.Vector2(150, self.HEIGHT - 40 - self.PLAYER_HEIGHT)
        self.player_vel = pygame.Vector2(0, 0)
        self.game_over = False

    def _check_cleared_obstacles(self):
        reward = 0
        player_center_x = self.player_rect.centerx
        for i, obstacle in enumerate(self.obstacles):
            if i not in self.obstacles_cleared and player_center_x > obstacle.right:
                # Check if player was above it previously
                if self.player_rect.bottom < obstacle.top + 10:
                    reward += 1.0 # Reward for jumping over
                    self.obstacles_cleared.add(i)
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update camera
        self.camera_x = self.player_pos.x - self.WIDTH / 2
        self.camera_x = max(0, self.camera_x)

        # Draw background grid
        grid_size = 50
        start_x = int(-self.camera_x % grid_size)
        for x in range(start_x, self.WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw obstacles
        for obstacle in self.obstacles:
            view_rect = obstacle.move(-self.camera_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, view_rect)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_OBSTACLE), view_rect, 2)

        # Draw finish line
        finish_rect = pygame.Rect(self.finish_line_x - self.camera_x, 0, 20, self.HEIGHT)
        if finish_rect.colliderect(self.screen.get_rect()):
            for i in range(0, self.HEIGHT, 20):
                color1 = self.COLOR_FINISH if (i // 20) % 2 == 0 else (0,0,0)
                color2 = (0,0,0) if (i // 20) % 2 == 0 else self.COLOR_FINISH
                pygame.draw.rect(self.screen, color1, (finish_rect.x, i, 10, 20))
                pygame.draw.rect(self.screen, color2, (finish_rect.x + 10, i, 10, 20))

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

        # Draw player
        view_player_rect = self.player_rect.move(-self.camera_x, 0)
        
        # Glow effect
        glow_radius = int(self.PLAYER_WIDTH * 1.2)
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(s, (view_player_rect.centerx - glow_radius, view_player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, view_player_rect, border_radius=3)
        
        # Player eye
        eye_x = view_player_rect.centerx + (5 if self.player_vel.x >= 0 else -5)
        eye_y = view_player_rect.centery - 5
        pygame.draw.circle(self.screen, (255,255,255), (eye_x, eye_y), 3)
        pygame.draw.circle(self.screen, (0,0,0), (eye_x, eye_y), 1)

    def _render_ui(self):
        # Stage
        stage_text = self.font_small.render(f"STAGE {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

        # Timer
        time_str = f"{max(0, self.stage_time):.1f}"
        timer_color = self.COLOR_TIMER if self.stage_time > 10 else self.COLOR_OBSTACLE
        timer_text = self.font_large.render(time_str, True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": self.stage_time,
        }
        
    def _spawn_particles(self, count, pos, color, style):
        for _ in range(count):
            if style == 'up':
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(1, 4))
            elif style == 'side':
                angle = self.np_random.uniform(math.pi, 2*math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.uniform(10, 20),
                'color': color,
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and visualization.
    # It will not run in a headless environment.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "dummy", etc.
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Robot Platformer")
    except pygame.error:
        print("Pygame display could not be initialized. Running headlessly.")
        screen = None

    obs, info = env.reset()
    done = False
    
    # Main game loop
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_UP]:
            action[0] = 1

        if keys[pygame.K_r]: # Reset on 'r' key
            obs, info = env.reset()
            total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if screen:
            # Transpose the observation back to pygame's (width, height, channels) format
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(GameEnv.FPS)

    env.close()