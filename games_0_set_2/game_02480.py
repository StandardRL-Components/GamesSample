
# Generated: 2025-08-27T20:29:38.612122
# Source Brief: brief_02480.md
# Brief Index: 2480

        
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
        "Controls: Use arrow keys to guide the worm. Avoid the red predators and reach the blue exit portal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a worm through a procedurally generated isometric maze, dodging predators to reach the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_WORM_HEAD = (100, 255, 100)
    COLOR_WORM_BODY = (50, 200, 50)
    COLOR_WORM_OUTLINE = (20, 80, 20)
    COLOR_PREDATOR = (255, 50, 50)
    COLOR_PREDATOR_OUTLINE = (100, 20, 20)
    COLOR_EXIT_MAIN = (100, 150, 255)
    COLOR_EXIT_GLOW = (50, 80, 200)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TIMER_WARN = (255, 200, 0)

    # Grid
    GRID_W = 40
    GRID_H = 30
    TILE_W = 20
    TILE_H = 10
    
    # Game settings
    MAX_STAGES = 3
    STAGE_DURATION_SECONDS = 60
    WORM_LENGTH = 10
    WORM_RADIUS = 6
    PREDATOR_RADIUS = 8
    PREDATOR_BASE_SPEED = 1.5
    PREDATOR_SPEED_INCREASE = 0.1
    PREDATOR_SPEED_INTERVAL = 1000 # steps
    PREDATOR_SIGHT_RANGE = 200
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        
        # Internal state variables
        self.worm_segments = []
        self.predators = []
        self.exit_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.timer = 0
        self.total_distance_traveled = 0
        self.last_dist_to_exit = 0.0
        self.game_over_reason = ""
        self.terminated = False

        # Random number generator
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.terminated = False
        self.game_over_reason = ""
        self.total_distance_traveled = 0

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        reward = 0
        self.terminated = False
        
        # Update timer
        self.timer -= 1
        if self.timer <= 0:
            self.terminated = True
            self.game_over_reason = "TIME UP"
            reward -= 10
            # sfx: time_up_sound

        # Update game state only if not terminated
        if not self.terminated:
            self._update_worm(movement)
            self._update_predators()
            
            collision_reward, collision_terminated = self._check_collisions()
            reward += collision_reward
            if collision_terminated:
                self.terminated = True

        # Calculate distance-based reward
        reward += self._calculate_reward()

        self.score += reward
        self.steps += 1
        
        # Auto advance frame rate
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _setup_stage(self):
        # Reset timer for the stage
        self.timer = self.STAGE_DURATION_SECONDS * self.FPS

        # Place worm
        start_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.worm_segments = [start_pos] * self.WORM_LENGTH
        
        # Place exit
        while True:
            self.exit_pos = (
                self.np_random.integers(1, self.GRID_W - 1),
                self.np_random.integers(1, self.GRID_H - 1),
            )
            dist = math.hypot(self.exit_pos[0] - start_pos[0], self.exit_pos[1] - start_pos[1])
            if dist > 10: # Ensure exit is not too close
                break
        
        # Place predators
        self.predators = []
        for _ in range(self.stage):
            while True:
                pos = (
                    self.np_random.integers(1, self.GRID_W - 1),
                    self.np_random.integers(1, self.GRID_H - 1),
                )
                dist_to_worm = math.hypot(pos[0] - start_pos[0], pos[1] - start_pos[1])
                dist_to_exit = math.hypot(pos[0] - self.exit_pos[0], pos[1] - self.exit_pos[1])
                if dist_to_worm > 8 and dist_to_exit > 8: # Don't spawn on top of worm/exit
                    screen_pos = self._iso_to_screen(pos[0], pos[1])
                    self.predators.append({
                        "pos": list(screen_pos),
                        "patrol_target": list(self._iso_to_screen(
                            self.np_random.integers(1, self.GRID_W - 1),
                            self.np_random.integers(1, self.GRID_H - 1)
                        )),
                    })
                    break
        
        # Reset distance calculation
        head_pos = self.worm_segments[0]
        self.last_dist_to_exit = math.hypot(
            head_pos[0] - self.exit_pos[0], head_pos[1] - self.exit_pos[1]
        )

    def _update_worm(self, movement):
        head = self.worm_segments[0]
        new_head = head
        
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            new_head = (head[0] - 1, head[1] - 1)
        elif movement == 2: # Down
            new_head = (head[0] + 1, head[1] + 1)
        elif movement == 3: # Left
            new_head = (head[0] - 1, head[1] + 1)
        elif movement == 4: # Right
            new_head = (head[0] + 1, head[1] - 1)
        
        if new_head != head:
            # Check boundaries
            if not (0 <= new_head[0] < self.GRID_W and 0 <= new_head[1] < self.GRID_H):
                new_head = head # Hit wall, don't move
                # sfx: bump_wall
            # Check self-collision
            elif new_head in self.worm_segments[1:]:
                self.terminated = True
                self.game_over_reason = "SELF-COLLISION"
                # sfx: self_destruct
                return # Stop update on termination

            self.worm_segments.insert(0, new_head)
            self.worm_segments.pop()
            self.total_distance_traveled += 1

    def _update_predators(self):
        worm_head_screen_pos = self._iso_to_screen(*self.worm_segments[0])
        speed = self.PREDATOR_BASE_SPEED + (self.steps // self.PREDATOR_SPEED_INTERVAL) * self.PREDATOR_SPEED_INCREASE

        for p in self.predators:
            dist_to_worm = math.hypot(p['pos'][0] - worm_head_screen_pos[0], p['pos'][1] - worm_head_screen_pos[1])
            
            target_pos = None
            if dist_to_worm < self.PREDATOR_SIGHT_RANGE:
                # Hunt worm
                target_pos = worm_head_screen_pos
            else:
                # Patrol
                dist_to_patrol = math.hypot(p['pos'][0] - p['patrol_target'][0], p['pos'][1] - p['patrol_target'][1])
                if dist_to_patrol < 20:
                    # New patrol target
                    new_grid_target = (
                        self.np_random.integers(1, self.GRID_W - 1),
                        self.np_random.integers(1, self.GRID_H - 1)
                    )
                    p['patrol_target'] = list(self._iso_to_screen(*new_grid_target))
                target_pos = p['patrol_target']

            # Move towards target
            dx = target_pos[0] - p['pos'][0]
            dy = target_pos[1] - p['pos'][1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                p['pos'][0] += (dx / dist) * speed
                p['pos'][1] += (dy / dist) * speed

    def _check_collisions(self):
        reward = 0
        terminated = False
        worm_head_grid = self.worm_segments[0]
        worm_head_screen = self._iso_to_screen(*worm_head_grid)

        # Worm vs Exit
        if worm_head_grid == self.exit_pos:
            # sfx: stage_complete_jingle
            self.score += 50 # Brief specified +50 for stage completion
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self._setup_stage()
            else:
                self.game_over_reason = "YOU WIN!"
                terminated = True
            return 50, terminated

        # Worm vs Predators
        for p in self.predators:
            dist = math.hypot(worm_head_screen[0] - p['pos'][0], worm_head_screen[1] - p['pos'][1])
            if dist < self.WORM_RADIUS + self.PREDATOR_RADIUS:
                # sfx: player_hit_sound
                self.game_over_reason = "CAUGHT!"
                return -10, True
        
        return reward, terminated

    def _calculate_reward(self):
        head_pos = self.worm_segments[0]
        current_dist = math.hypot(head_pos[0] - self.exit_pos[0], head_pos[1] - self.exit_pos[1])
        
        reward = 0
        if current_dist < self.last_dist_to_exit:
            reward = 0.1 # Closer to exit
        elif current_dist > self.last_dist_to_exit:
            reward = -0.2 # Further from exit
            
        self.last_dist_to_exit = current_dist
        return reward

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = (self.SCREEN_WIDTH / 2) + (grid_x - grid_y) * (self.TILE_W / 2)
        screen_y = (self.SCREEN_HEIGHT / 4) + (grid_x + grid_y) * (self.TILE_H / 2)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_W + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_H)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.GRID_H + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_W, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw exit portal
        exit_screen_pos = self._iso_to_screen(*self.exit_pos)
        pulse = abs(math.sin(self.steps * 0.1))
        for i in range(5, 0, -1):
            radius = int(self.TILE_W * (0.5 + pulse * 0.1) + i * 2)
            alpha = 150 - i * 30
            color = (*self.COLOR_EXIT_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, exit_screen_pos[0], exit_screen_pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, exit_screen_pos[0], exit_screen_pos[1], int(self.TILE_W * 0.5), self.COLOR_EXIT_MAIN)
        
        # Draw worm
        for i, seg in reversed(list(enumerate(self.worm_segments))):
            pos = self._iso_to_screen(*seg)
            radius = self.WORM_RADIUS
            color = self.COLOR_WORM_BODY
            if i == 0: # Head
                radius = self.WORM_RADIUS + 2
                color = self.COLOR_WORM_HEAD
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_WORM_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Draw predators
        for p in self.predators:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PREDATOR_RADIUS, self.COLOR_PREDATOR_OUTLINE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PREDATOR_RADIUS, self.COLOR_PREDATOR)
            
    def _render_ui(self):
        # Stage
        stage_text = self.font_small.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Distance
        dist_text = self.font_small.render(f"Distance: {self.total_distance_traveled}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (self.SCREEN_WIDTH - dist_text.get_width() - 10, 10))
        
        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_TEXT
        if time_left < 10:
            timer_color = self.COLOR_TIMER_WARN
            if self.steps % self.FPS < self.FPS / 2: # Blink effect
                 timer_color = self.COLOR_PREDATOR
        
        timer_text = self.font_large.render(f"{time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH/2 - timer_text.get_width()/2, self.SCREEN_HEIGHT - timer_text.get_height() - 5))

        # Game Over Screen
        if self.terminated:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_over_reason, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "timer": round(self.timer / self.FPS, 2),
            "distance_traveled": self.total_distance_traveled,
            "is_terminated": self.terminated,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy window to display the game
    pygame.display.set_caption("Isometric Worm")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Get user input
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()