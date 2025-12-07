
# Generated: 2025-08-27T23:58:58.850641
# Source Brief: brief_03648.md
# Brief Index: 3648

        
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
        "Controls: ↑ to move up, ↓ to move down. Dodge the red obstacles and reach the finish line."
    )

    game_description = (
        "A fast-paced line racer. Dodge procedurally generated obstacles and race to the finish line against the clock."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_TRACK = (60, 20, 80)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_OUTLINE = (255, 150, 150)
        self.COLOR_FINISH_LINE = (50, 255, 50)
        self.COLOR_PARTICLE = (255, 200, 100)
        self.COLOR_TEXT = (220, 220, 240)
        
        # Game constants
        self.MAX_STEPS = 1000
        self.FPS = 30
        self.PLAYER_HORIZ_SPEED = 5
        self.PLAYER_VERT_SPEED = 6
        self.PLAYER_WIDTH = 15
        self.PLAYER_HEIGHT = 15
        
        self.TRACK_Y_TOP = 80
        self.TRACK_Y_BOTTOM = self.HEIGHT - 80
        self.TRACK_HEIGHT = self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP

        self.FINISH_LINE_X = self.PLAYER_HORIZ_SPEED * self.MAX_STEPS
        
        self.INITIAL_SPAWN_RATE = 0.05  # obstacles per step
        self.SPAWN_RATE_INCREASE = (0.005 / self.FPS) # per step

        # Initialize state variables
        self.player_pos = None
        self.world_scroll_x = None
        self.obstacles = None
        self.particles = None
        self.current_spawn_rate = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [100, self.HEIGHT / 2]
        self.world_scroll_x = 0.0
        self.obstacles = []
        self.particles = []
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_world()
        self._spawn_obstacles()
        
        reward, terminated = self._check_collisions_and_rewards()
        self.score += reward
        self.game_over = terminated

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.win:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_VERT_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_VERT_SPEED
        
        # Clamp player position within track boundaries
        self.player_pos[1] = np.clip(
            self.player_pos[1], 
            self.TRACK_Y_TOP + self.PLAYER_HEIGHT, 
            self.TRACK_Y_BOTTOM - self.PLAYER_HEIGHT
        )

    def _update_world(self):
        # Scroll the world
        self.world_scroll_x += self.PLAYER_HORIZ_SPEED
        
        # Update spawn rate
        self.current_spawn_rate += self.SPAWN_RATE_INCREASE

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] + obs['w'] > self.world_scroll_x]

    def _spawn_obstacles(self):
        if self.np_random.random() < self.current_spawn_rate:
            min_gap = self.PLAYER_HEIGHT * 2.5
            max_gap = self.PLAYER_HEIGHT * 4.0
            gap_size = self.np_random.uniform(min_gap, max_gap)
            gap_y = self.np_random.uniform(
                self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM - gap_size
            )
            
            obstacle_width = self.np_random.integers(20, 50)
            spawn_x = self.world_scroll_x + self.WIDTH + 50
            
            # Top obstacle
            h1 = gap_y - self.TRACK_Y_TOP
            if h1 > 5:
                self.obstacles.append({'x': spawn_x, 'y': self.TRACK_Y_TOP, 'w': obstacle_width, 'h': h1})

            # Bottom obstacle
            h2 = self.TRACK_Y_BOTTOM - (gap_y + gap_size)
            if h2 > 5:
                self.obstacles.append({'x': spawn_x, 'y': gap_y + gap_size, 'w': obstacle_width, 'h': h2})
            
            # sfx: obstacle_spawn.wav

    def _check_collisions_and_rewards(self):
        reward = 0.1  # Survival reward
        terminated = False
        
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_WIDTH / 2,
            self.player_pos[1] - self.PLAYER_HEIGHT / 2,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )

        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'] - self.world_scroll_x, obs['y'], obs['w'], obs['h'])
            
            # Check for actual collision
            if player_rect.colliderect(obs_rect):
                # sfx: player_explosion.wav
                reward = -100
                terminated = True
                self.win = False
                return reward, terminated

            # Check for near miss
            near_miss_rect = obs_rect.inflate(self.PLAYER_WIDTH * 2, self.PLAYER_HEIGHT * 2)
            if near_miss_rect.colliderect(player_rect):
                reward -= 5
                self._create_particles(player_rect.center, 5)
                # sfx: near_miss.wav

        # Check for win condition
        if self.player_pos[0] + self.world_scroll_x >= self.FINISH_LINE_X:
            reward = 100
            terminated = True
            self.win = True
            # sfx: win_jingle.wav
            
        return reward, terminated

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(1, 3)
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw track boundaries
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.WIDTH, self.TRACK_Y_TOP), 3)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.WIDTH, self.TRACK_Y_BOTTOM), 3)

        # Draw finish line
        finish_screen_x = self.FINISH_LINE_X - self.world_scroll_x
        if 0 < finish_screen_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_screen_x, self.TRACK_Y_TOP), (finish_screen_x, self.TRACK_Y_BOTTOM), 5)

        # Draw obstacles
        for obs in self.obstacles:
            screen_x = obs['x'] - self.world_scroll_x
            rect = (int(screen_x), int(obs['y']), int(obs['w']), int(obs['h']))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, rect, 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = self.COLOR_PARTICLE + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Draw player
        if not (self.game_over and not self.win): # Don't draw player if crashed
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            p1 = (px + self.PLAYER_WIDTH, py)
            p2 = (px - self.PLAYER_WIDTH // 2, py - self.PLAYER_HEIGHT // 2)
            p3 = (px - self.PLAYER_WIDTH // 2, py + self.PLAYER_HEIGHT // 2)
            
            # Glow effect
            glow_surface = pygame.Surface((self.PLAYER_WIDTH * 4, self.PLAYER_WIDTH * 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_trigon(glow_surface, 
                                        self.PLAYER_WIDTH*2, self.PLAYER_WIDTH*2,
                                        0, self.PLAYER_WIDTH,
                                        0, self.PLAYER_WIDTH*3,
                                        self.COLOR_PLAYER_GLOW)
            self.screen.blit(glow_surface, (px - self.PLAYER_WIDTH*2 + self.PLAYER_WIDTH/2, py - self.PLAYER_WIDTH*2))
            
            # Main ship
            pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Time/Steps left
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Game Over message
        if self.game_over:
            msg = "FINISH!" if self.win else "GAME OVER"
            color = self.COLOR_FINISH_LINE if self.win else self.COLOR_OBSTACLE
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    
    # Use Pygame for human interaction
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    # --- Main Game Loop ---
    while running:
        if terminated:
            # Wait for a moment on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Event Handling ---
        action = env.action_space.sample() # Start with a random action
        action[0] = 0 # Default to no-op for movement
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                 obs, info = env.reset()
                 terminated = False

        # --- Player Controls ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Frame Rate ---
        env.clock.tick(env.FPS)

    env.close()