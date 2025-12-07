
# Generated: 2025-08-27T23:28:24.619401
# Source Brief: brief_03482.md
# Brief Index: 3482

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist, top-down block breaker where an agent controls a paddle
    to destroy falling blocks. The game prioritizes visual feedback and a
    satisfying, arcade-like feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use ← and → to move the paddle left and right to intercept falling blocks."
    )

    # User-facing game description
    game_description = (
        "A minimalist, top-down block breaker. Intercept falling blocks with your paddle to score points. "
        "The game gets faster as you destroy more blocks. Don't let any blocks reach the bottom!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (180, 180, 200)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_BLOCK_SLOW = (100, 150, 255)
    COLOR_BLOCK_FAST = (50, 100, 255)
    
    # Paddle
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    PADDLE_Y_POS = SCREEN_HEIGHT - 40
    
    # Block
    BLOCK_WIDTH = 40
    BLOCK_HEIGHT = 20
    INITIAL_BLOCK_SPEED = 2.0
    SPEED_INCREASE_RATE = 0.2
    BLOCKS_PER_SPEED_INCREASE = 10
    
    # Game
    WIN_CONDITION_BLOCKS = 50
    MAX_STEPS = 1500 # Approx 50 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 24, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.blocks_destroyed = 0
        self.current_block_speed = self.INITIAL_BLOCK_SPEED
        self.game_over = False
        
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.blocks_destroyed = 0
        self.game_over = False
        self.current_block_speed = self.INITIAL_BLOCK_SPEED
        
        # Create paddle
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, self.PADDLE_Y_POS, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Clear dynamic elements
        self.blocks.clear()
        self.particles.clear()
        
        # Spawn initial block
        self._spawn_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Small penalty per step to encourage efficiency
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # --- 1. Update Player (Paddle) ---
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen boundaries
            self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

            # --- 2. Update Game Logic ---
            self._update_blocks()
            self._update_particles()
            
            # Spawn a new block if the screen is empty
            if not self.blocks:
                self._spawn_block()

            # --- 3. Calculate Rewards from block updates ---
            # This is handled within _update_blocks and assigned to self.score
            # The reward signal is constructed based on game events
            new_reward, terminated_by_event = self._process_events()
            reward += new_reward
            if terminated_by_event:
                self.game_over = True
        
        # --- 4. Check Termination Conditions ---
        self.steps += 1
        terminated = self.game_over
        if self.blocks_destroyed >= self.WIN_CONDITION_BLOCKS:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _spawn_block(self):
        x_pos = self.np_random.integers(0, self.SCREEN_WIDTH - self.BLOCK_WIDTH)
        # 30% chance of a fast block
        is_fast = self.np_random.random() < 0.3
        
        block_type = 'fast' if is_fast else 'slow'
        speed = self.current_block_speed * 1.5 if is_fast else self.current_block_speed
        
        block_rect = pygame.Rect(x_pos, -self.BLOCK_HEIGHT, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        self.blocks.append({'rect': block_rect, 'type': block_type, 'speed': speed})

    def _update_blocks(self):
        # We store events and process them later to build the reward signal
        self.events = []
        for block in self.blocks[:]:
            block['rect'].y += block['speed']
            
            # Paddle collision
            if block['rect'].colliderect(self.paddle):
                # SFX: paddle_hit.wav
                self.events.append(('block_destroyed', block))
                self.blocks.remove(block)
                self._create_particles(block['rect'].midbottom, block['type'])
                continue

            # Bottom wall collision (loss)
            if block['rect'].top > self.SCREEN_HEIGHT:
                # SFX: game_over.wav
                self.events.append(('game_over',))
                self.blocks.remove(block)
                continue

    def _process_events(self):
        reward = 0
        terminated = False
        for event in self.events:
            event_type = event[0]
            if event_type == 'block_destroyed':
                block_data = event[1]
                
                # Base reward for destroying block
                block_reward = 2.0 if block_data['type'] == 'fast' else 1.0
                
                # Bonus reward for risky edge hits
                dist_from_center = abs(block_data['rect'].centerx - self.paddle.centerx)
                if dist_from_center > self.PADDLE_WIDTH / 2 - 10:
                    block_reward += 1.0 # Risky play bonus
                
                reward += block_reward
                self.score += int(block_reward)
                self.blocks_destroyed += 1

                # Increase difficulty
                if self.blocks_destroyed > 0 and self.blocks_destroyed % self.BLOCKS_PER_SPEED_INCREASE == 0:
                    self.current_block_speed += self.SPEED_INCREASE_RATE
                    # SFX: level_up.wav
            
            elif event_type == 'game_over':
                reward -= 100
                terminated = True
                
        return reward, terminated

    def _create_particles(self, position, block_type):
        color = self.COLOR_BLOCK_FAST if block_type == 'fast' else self.COLOR_BLOCK_SLOW
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(3, 7)
            self.particles.append({
                'pos': list(position),
                'vel': velocity,
                'life': lifetime,
                'max_life': lifetime,
                'color': color,
                'radius': radius
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render walls
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (self.SCREEN_WIDTH, 0), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (0, self.SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - 1, 0), (self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT), 2)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            current_radius = int(p['radius'] * (p['life'] / p['max_life']))
            if current_radius > 0:
                color = (*p['color'], alpha)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), current_radius, color)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), current_radius, color)

        # Render blocks
        for block in self.blocks:
            color = self.COLOR_BLOCK_FAST if block['type'] == 'fast' else self.COLOR_BLOCK_SLOW
            pygame.draw.rect(self.screen, color, block['rect'], border_radius=3)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render UI
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        blocks_text = self.font.render(f"DESTROYED: {self.blocks_destroyed}/{self.WIN_CONDITION_BLOCKS}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.SCREEN_WIDTH - blocks_text.get_width() - 10, 10))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_destroyed": self.blocks_destroyed,
            "current_block_speed": self.current_block_speed,
        }
        
    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # This game doesn't use space or shift, but we must provide the actions
        space_held = 0
        shift_held = 0
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # --- Frame Rate ---
        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    
    env.close()