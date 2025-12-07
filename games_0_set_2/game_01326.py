
# Generated: 2025-08-27T16:46:28.880082
# Source Brief: brief_01326.md
# Brief Index: 1326

        
import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"  # Pygame runs headless

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Avoid the red blocks and reach the green finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade dodger. Navigate your ship through a hazardous, procedurally generated field of moving blocks to reach the finish line before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (100, 200, 255)
        self.COLOR_BLOCK = (255, 50, 50)
        self.COLOR_BLOCK_OUTLINE = (150, 20, 20)
        self.COLOR_FINISH = (0, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 200, 0)
        
        # Game parameters
        self.PLAYER_SPEED = 5.0
        self.PLAYER_SIZE = 12
        self.BLOCK_SIZE = 25
        self.NUM_BLOCKS = 30
        self.MAX_STEPS = 1800 # 60 seconds at 30fps, but using steps for termination
        self.FINISH_LINE_X = self.WIDTH - 20
        self.DIFFICULTY_INTERVAL = 200
        self.MAX_BLOCK_SPEED = 5.0
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.base_block_speed = None
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.math.Vector2(50, self.HEIGHT // 2)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_block_speed = 1.0
        
        self.blocks = []
        for _ in range(self.NUM_BLOCKS):
            self._spawn_block(initial_spawn=True)

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        
        # 1. Update Player
        movement_vec = self._update_player(movement)
        
        # 2. Update Blocks and Difficulty
        self._update_difficulty()
        self._update_blocks()
        
        # 3. Update Particles
        self._update_particles()
        
        # 4. Check for termination conditions
        terminated, win = self._check_termination()
        self.game_over = terminated
        self.game_won = win
        
        # 5. Calculate reward
        reward = self._calculate_reward(movement_vec, terminated, win)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_block(self, initial_spawn=False):
        # Spawn away from the player's start area on initial spawn
        x_pos = self.np_random.integers(self.WIDTH // 2 if initial_spawn else 0, self.WIDTH)
        y_pos = self.np_random.integers(-self.HEIGHT, 0)
        rect = pygame.Rect(x_pos, y_pos, self.BLOCK_SIZE, self.BLOCK_SIZE)
        speed = self.base_block_speed + self.np_random.uniform(-0.2, 0.2)
        self.blocks.append({'rect': rect, 'speed': max(0.5, speed)})

    def _update_player(self, movement):
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
        elif movement == 2:  # Down
            move_vec.y = 1
        elif movement == 3:  # Left
            move_vec.x = -1
        elif movement == 4:  # Right
            move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.scale_to_length(self.PLAYER_SPEED)
            self.player_pos += move_vec
            self._create_particles(self.player_pos, -move_vec)

        # Clamp player position to screen bounds
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.WIDTH - self.PLAYER_SIZE))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.player_pos.y, self.HEIGHT - self.PLAYER_SIZE))
        return move_vec

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.base_block_speed = min(self.MAX_BLOCK_SPEED, self.base_block_speed + 0.05)

    def _update_blocks(self):
        for block in self.blocks:
            block['rect'].y += block['speed']
            if block['rect'].top > self.HEIGHT:
                block['rect'].x = self.np_random.integers(0, self.WIDTH - self.BLOCK_SIZE)
                block['rect'].y = self.np_random.integers(-self.BLOCK_SIZE * 5, -self.BLOCK_SIZE)
                block['speed'] = self.base_block_speed + self.np_random.uniform(-0.2, 0.2)

    def _create_particles(self, pos, base_vel):
        for _ in range(3):
            particle_pos = pos.copy()
            angle = base_vel.angle_to(pygame.math.Vector2(1,0)) + self.np_random.uniform(-30, 30)
            speed = self.np_random.uniform(1, 2)
            particle_vel = pygame.math.Vector2()
            particle_vel.from_polar((speed, angle))
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({'pos': particle_pos, 'vel': particle_vel, 'life': lifetime})
            # Sfx placeholder: player_thrust.wav

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_termination(self):
        # 1. Win condition
        if self.player_pos.x > self.FINISH_LINE_X:
            return True, True  # terminated, won

        # 2. Collision with blocks
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for block in self.blocks:
            if player_rect.colliderect(block['rect']):
                # Sfx placeholder: player_explosion.wav
                return True, False  # terminated, lost

        # 3. Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True, False  # terminated, lost
        
        return False, False # not terminated, not won

    def _calculate_reward(self, movement_vec, terminated, win):
        if terminated:
            if win:
                # Sfx placeholder: level_win.wav
                speed_bonus = (self.MAX_STEPS - self.steps) / 10.0
                return 100.0 + speed_bonus
            else: # Lost by collision or time out
                return -10.0
        
        reward = 0.0
        # Survival reward
        reward += 0.01

        # Penalty for moving away from the goal
        if movement_vec.x < 0:
            reward -= 0.005

        return reward

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
        # Draw finish line
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (self.FINISH_LINE_X, 0, self.WIDTH - self.FINISH_LINE_X, self.HEIGHT))
        
        # Draw particles
        for p in self.particles:
            size = max(1, int(p['life'] * 0.2))
            pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p['pos'].x), int(p['pos'].y)), size)
            
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_OUTLINE, block['rect'])
            inner_rect = block['rect'].inflate(-4, -4)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, inner_rect)

        # Draw player ship (triangle)
        p1 = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE, 0)
        p2 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE / 2, -self.PLAYER_SIZE * 0.866)
        p3 = self.player_pos + pygame.math.Vector2(-self.PLAYER_SIZE / 2, self.PLAYER_SIZE * 0.866)
        points = [(int(p.x), int(p.y)) for p in [p1, p2, p3]]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_seconds = time_left / 30.0 # Assuming ~30 steps/sec for display purposes
        time_text = self.font_ui.render(f"Time: {time_seconds:.1f}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over / Win message
        if self.game_over:
            message = "YOU WON!" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_BLOCK
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_won": self.game_won,
        }
        
    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To display the game, we need to set up a display
    pygame.display.set_caption("Arcade Dodger")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    while running:
        action = 0 # Default action: no-op
        
        # Manual control for human play
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3
        elif keys[pygame.K_RIGHT]:
            action = 4
        
        # The action space is MultiDiscrete, so we create the full action array
        full_action = [action, 0, 0]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(full_action)
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and terminated:
                obs, info = env.reset()
                terminated = False
                
        env.clock.tick(30) # Limit frame rate for human play

    env.close()