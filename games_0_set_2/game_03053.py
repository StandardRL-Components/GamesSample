
# Generated: 2025-08-28T06:51:48.091568
# Source Brief: brief_03053.md
# Brief Index: 3053

        
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
        "Controls: ↑ to move up, ↓ to move down. Dodge the red zombies and collect pink hearts."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Dodge hordes of procedurally generated zombies in a side-scrolling arcade environment to survive for 60 seconds."
    )

    # Frames auto-advance at 60fps for this real-time game.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_HEART = (255, 105, 180)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Game constants
        self.FPS = 60
        self.WIN_TIME_SECONDS = 60.0
        
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5
        self.ZOMBIE_SIZE = 20
        self.HEART_RADIUS = 10
        
        self.INITIAL_ZOMBIE_SPEED = 2.0
        self.ZOMBIE_SPAWN_RATE_INITIAL = 45 # frames
        self.ZOMBIE_SPAWN_RATE_MIN = 15
        self.HEART_SPAWN_RATE = 450 # frames
        
        # Initialize state variables
        self.player_pos = None
        self.zombies = None
        self.hearts = None
        self.time_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.zombie_speed = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_rate = None
        self.heart_spawn_timer = None

        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for submission, but useful for testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = pygame.Vector2(50, self.HEIGHT / 2)
        self.zombies = []
        self.hearts = []
        
        self.time_remaining = self.WIN_TIME_SECONDS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.zombie_spawn_rate = self.ZOMBIE_SPAWN_RATE_INITIAL
        self.zombie_spawn_timer = self.zombie_spawn_rate
        self.heart_spawn_timer = self.HEART_SPAWN_RATE
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Small reward for surviving a frame
        
        # Unpack factorized action
        movement = action[0]
        
        # --- PLAYER LOGIC ---
        if movement == 1:  # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos.y += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.y = max(self.PLAYER_SIZE / 2, min(self.HEIGHT - self.PLAYER_SIZE / 2, self.player_pos.y))
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # --- GAME STATE UPDATES ---
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # --- DIFFICULTY SCALING ---
        # Increase zombie speed every 10 seconds
        if self.steps > 0 and self.steps % (self.FPS * 10) == 0:
            self.zombie_speed += 0.5
        # Increase zombie spawn rate over time
        self.zombie_spawn_rate = max(self.ZOMBIE_SPAWN_RATE_MIN, self.ZOMBIE_SPAWN_RATE_INITIAL - (self.steps / 120))

        # --- SPAWNING ---
        self._spawn_zombies()
        self._spawn_hearts()
        
        # --- ENTITY UPDATES & COLLISIONS ---
        self._update_zombies(player_rect)
        reward += self._update_hearts(player_rect)
        
        # --- TERMINATION CHECK ---
        terminated = self.game_over
        if not terminated and self.time_remaining <= 0:
            self.time_remaining = 0
            reward = 50.0  # Large reward for winning
            terminated = True
            self.game_over = True
            self.win = True
            # Sound effect: win
        
        if self.game_over and not self.win:
            reward = -10.0 # Large penalty for losing
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            y_pos = self.np_random.integers(self.ZOMBIE_SIZE // 2, self.HEIGHT - self.ZOMBIE_SIZE // 2)
            zombie_rect = pygame.Rect(self.WIDTH, y_pos - self.ZOMBIE_SIZE // 2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            self.zombies.append(zombie_rect)
            self.zombie_spawn_timer = self.np_random.integers(int(self.zombie_spawn_rate * 0.8), int(self.zombie_spawn_rate * 1.2))

    def _spawn_hearts(self):
        self.heart_spawn_timer -= 1
        if self.heart_spawn_timer <= 0 and len(self.hearts) < 2:
            x_pos = self.np_random.integers(100, self.WIDTH - 100)
            y_pos = self.np_random.integers(50, self.HEIGHT - 50)
            heart = {
                'pos': pygame.Vector2(x_pos, y_pos),
                'rect': pygame.Rect(x_pos - self.HEART_RADIUS, y_pos - self.HEART_RADIUS, self.HEART_RADIUS*2, self.HEART_RADIUS*2),
                'spawn_time': self.steps
            }
            self.hearts.append(heart)
            self.heart_spawn_timer = self.HEART_SPAWN_RATE

    def _update_zombies(self, player_rect):
        for zombie in self.zombies[:]:
            zombie.x -= self.zombie_speed
            if zombie.right < 0:
                self.zombies.remove(zombie)
            elif player_rect.colliderect(zombie):
                self.game_over = True
                # Sound effect: player_hit
    
    def _update_hearts(self, player_rect):
        reward = 0
        for heart in self.hearts[:]:
            if player_rect.colliderect(heart['rect']):
                self.hearts.remove(heart)
                self.score += 1
                self.time_remaining += 2.0
                reward += 1.0
                # Sound effect: collect
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render hearts
        for heart in self.hearts:
            # Blinking effect using a sine wave on the radius
            blink_phase = (self.steps - heart['spawn_time']) * 0.2
            radius = self.HEART_RADIUS * (0.85 + 0.15 * math.sin(blink_phase))
            pygame.gfxdraw.aacircle(self.screen, int(heart['pos'].x), int(heart['pos'].y), int(radius), self.COLOR_HEART)
            pygame.gfxdraw.filled_circle(self.screen, int(heart['pos'].x), int(heart['pos'].y), int(radius), self.COLOR_HEART)

        # Render zombies
        for zombie in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie)

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Render game over/win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 50, 50)
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Render time remaining
        time_text = f"TIME: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Render score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Dodge")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        action = [movement, 0, 0] # Space and Shift are not used

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a bit after game over
    end_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - end_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        else:
            continue
        break

    env.close()