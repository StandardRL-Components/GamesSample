
# Generated: 2025-08-27T18:17:08.697710
# Source Brief: brief_01784.md
# Brief Index: 1784

        
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
        "Controls: Press SPACE to jump over obstacles. Try to reach the finish line!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot in a fast-paced, side-scrolling obstacle course. "
        "Jump to avoid hazards and reach the finish line in the fastest time possible."
    )

    # Frames auto-advance for smooth, time-based gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 30.0
        self.MAX_HITS = 3
        self.MAX_STEPS = self.FPS * int(self.GAME_DURATION_SECONDS + 5) # Time limit + buffer

        # Player constants
        self.PLAYER_SIZE = 24
        self.PLAYER_X_POS = self.WIDTH // 4
        self.GROUND_Y = self.HEIGHT - 50
        self.JUMP_VELOCITY = -20
        self.GRAVITY = 1.2

        # World constants
        self.WORLD_SCROLL_SPEED = 250  # pixels per second

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GROUND = (60, 60, 70)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_OBSTACLE = (255, 50, 100)
        self.COLOR_FINISH = (50, 255, 150)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_PARTICLE = (255, 255, 200)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Initialize state variables
        self.player_y = 0
        self.player_vy = 0
        self.on_ground = False
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.hits = 0
        self.distance_to_finish = 0
        self.next_obstacle_spawn_time = 0
        self.base_spawn_interval = 1.8
        self.last_jump_action = False
        self.win_condition_met = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.hits = 0
        self.win_condition_met = False

        # Player state
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.on_ground = True
        self.last_jump_action = False

        # World state
        self.distance_to_finish = self.WORLD_SCROLL_SPEED * self.GAME_DURATION_SECONDS
        self.obstacles = []
        self.particles = []
        self.next_obstacle_spawn_time = self.np_random.uniform(1.5, 2.5)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        space_held = action[1] == 1
        
        # --- Game Logic Update ---
        delta_time = 1.0 / self.FPS
        self.time_elapsed += delta_time
        reward = 0.1  # Survival reward

        # 1. Handle Input
        if space_held and not self.last_jump_action and self.on_ground:
            self.player_vy = self.JUMP_VELOCITY
            self.on_ground = False
            # Sound placeholder: jump_sound.play()
        self.last_jump_action = space_held

        # 2. Player Physics
        if not self.on_ground:
            self.player_vy += self.GRAVITY
            self.player_y += self.player_vy
            if self.player_y >= self.GROUND_Y:
                self.player_y = self.GROUND_Y
                self.player_vy = 0
                self.on_ground = True

        # 3. World Scrolling & Obstacle Management
        scroll_amount = self.WORLD_SCROLL_SPEED * delta_time
        self.distance_to_finish -= scroll_amount

        # Update and remove off-screen obstacles
        for obstacle in self.obstacles:
            obstacle['rect'].x -= scroll_amount
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        # Spawn new obstacles
        if self.time_elapsed > self.next_obstacle_spawn_time:
            obstacle_height = self.np_random.integers(40, 110)
            obstacle_rect = pygame.Rect(
                self.WIDTH,
                self.GROUND_Y - obstacle_height,
                self.np_random.integers(25, 40),
                obstacle_height
            )
            self.obstacles.append({'rect': obstacle_rect, 'hit': False})

            # Difficulty scaling
            time_factor = max(0, self.time_elapsed - 10)
            interval_reduction = time_factor * 0.08
            min_interval = 0.6
            current_interval = max(min_interval, self.base_spawn_interval - interval_reduction)
            self.next_obstacle_spawn_time = self.time_elapsed + self.np_random.uniform(current_interval * 0.8, current_interval * 1.2)

        # 4. Collision Detection
        player_rect = pygame.Rect(self.PLAYER_X_POS, self.player_y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obstacle in self.obstacles:
            if not obstacle['hit'] and player_rect.colliderect(obstacle['rect']):
                obstacle['hit'] = True
                self.hits += 1
                reward -= 5.0
                # Sound placeholder: collision_sound.play()
                # Particle burst on impact
                for _ in range(30):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(2, 8)
                    vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                    self.particles.append({
                        'pos': list(player_rect.center),
                        'vel': vel,
                        'life': self.np_random.integers(15, 30),
                        'radius': self.np_random.uniform(1, 4)
                    })

        # 5. Update Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2  # Gravity on particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # 6. Check Termination Conditions
        terminated = False
        if self.hits >= self.MAX_HITS:
            terminated = True
            # Sound placeholder: game_over_sound.play()
        
        if self.time_elapsed >= self.GAME_DURATION_SECONDS and self.distance_to_finish > 0:
            terminated = True # Time ran out
        
        if self.distance_to_finish <= 0 and not self.win_condition_met:
            self.win_condition_met = True
            terminated = True
            # Calculate win bonus
            time_bonus = 50 * max(0, (self.GAME_DURATION_SECONDS - self.time_elapsed) / self.GAME_DURATION_SECONDS)
            reward += time_bonus
            # Sound placeholder: win_sound.play()

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Draw finish line if in view
        finish_x = self.PLAYER_X_POS + self.distance_to_finish
        if finish_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.HEIGHT), 5)

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.gfxdraw.box(self.screen, obstacle['rect'], self.COLOR_OBSTACLE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Draw player
        player_rect = pygame.Rect(
            self.PLAYER_X_POS, 
            int(self.player_y - self.PLAYER_SIZE), 
            self.PLAYER_SIZE, 
            self.PLAYER_SIZE
        )
        # Glow effect
        glow_center = player_rect.center
        glow_radius = int(self.PLAYER_SIZE * 0.8)
        glow_color = (*self.COLOR_PLAYER_GLOW, 60) # RGBA for transparency
        pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], glow_radius, glow_color)
        # Player body
        pygame.gfxdraw.box(self.screen, player_rect, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time remaining
        time_text = f"Time: {max(0, self.GAME_DURATION_SECONDS - self.time_elapsed):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Hits
        hits_text = f"Hits: {self.hits} / {self.MAX_HITS}"
        hits_surf = self.font_ui.render(hits_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(hits_surf, (self.WIDTH - hits_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win_condition_met:
                msg = "FINISH!"
                color = self.COLOR_FINISH
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE
            
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.hits,
            "time_elapsed": self.time_elapsed,
            "distance_to_finish": self.distance_to_finish,
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space is held
        if keys[pygame.K_ESCAPE]:
            running = False
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart or ESC to quit.")
            
            # Wait for restart or quit
            pause = True
            while pause:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pause = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        pause = False
                clock.tick(env.FPS)

        clock.tick(env.FPS)
        
    env.close()