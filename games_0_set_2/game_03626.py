
# Generated: 2025-08-27T23:55:46.217861
# Source Brief: brief_03626.md
# Brief Index: 3626

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Survive the block onslaught and aim for a high score!"
    )

    # User-facing game description
    game_description = (
        "A fast-paced, retro arcade block-breaker. Deflect falling blocks with your paddle, "
        "grab power-ups, and survive as long as you can to clear all 100 blocks."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.TOTAL_BLOCKS = 100
        self.INITIAL_LIVES = 3
        
        # Colors
        self.COLOR_BG_TOP = (10, 5, 25)
        self.COLOR_BG_BOTTOM = (30, 15, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_FLASH = (200, 0, 0, 100)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255), 
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]
        
        # Gameplay
        self.PADDLE_SPEED = 8
        self.PADDLE_WIDTH_NORMAL = 80
        self.PADDLE_WIDTH_EXTENDED = 160
        self.PADDLE_HEIGHT = 12
        self.BLOCK_WIDTH = 40
        self.BLOCK_HEIGHT = 20
        self.POWERUP_SIZE = 15
        self.POWERUP_CHANCE = 0.15
        self.POWERUP_DURATION = 300 # 10 seconds at 30fps

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except FileNotFoundError:
            print("Default font not found, using fallback.")
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)

        # --- Initialize State ---
        self.paddle = None
        self.blocks = None
        self.particles = None
        self.powerup_items = None
        self.active_powerups = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.blocks_destroyed = None
        self.blocks_spawned = None
        self.fall_speed = None
        self.spawn_timer = None
        self.spawn_interval = None
        self.screen_flash_timer = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH_NORMAL // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH_NORMAL,
            self.PADDLE_HEIGHT
        )
        self.blocks = []
        self.particles = []
        self.powerup_items = []
        self.active_powerups = {}

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self.blocks_destroyed = 0
        self.blocks_spawned = 0
        self.fall_speed = 2.0
        self.spawn_timer = 0
        self.spawn_interval = 45 # Spawn a block every 1.5 seconds
        self.screen_flash_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Base Reward ---
        reward = -0.02 # Small penalty for existing
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_game_state()

        # --- Handle Collisions and Events ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Update Step Counter and Check Termination ---
        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True

        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.paddle.width))

    def _update_game_state(self):
        # Spawn new blocks
        if self.blocks_spawned < self.TOTAL_BLOCKS:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_timer = 0
                self._spawn_block()
        
        # Move blocks
        for block in self.blocks:
            block['rect'].y += self.fall_speed

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # Update power-up items
        for item in self.powerup_items:
            item['rect'].y += 1.5

        # Update active power-ups
        if 'extend_paddle' in self.active_powerups:
            self.active_powerups['extend_paddle'] -= 1
            if self.active_powerups['extend_paddle'] <= 0:
                del self.active_powerups['extend_paddle']
                center_x = self.paddle.centerx
                self.paddle.width = self.PADDLE_WIDTH_NORMAL
                self.paddle.centerx = center_x
        
        # Update screen flash
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1
            
    def _spawn_block(self):
        self.blocks_spawned += 1
        block_x = self.np_random.integers(0, self.WIDTH - self.BLOCK_WIDTH)
        block_color = random.choice(self.BLOCK_COLORS)
        self.blocks.append({
            'rect': pygame.Rect(block_x, -self.BLOCK_HEIGHT, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            'color': block_color
        })

    def _handle_collisions(self):
        reward = 0
        
        # Block collisions
        for block in self.blocks[:]:
            # Block hits paddle
            if self.paddle.colliderect(block['rect']):
                # sfx: paddle_hit.wav
                self.blocks.remove(block)
                self.blocks_destroyed += 1
                self.score += 10
                reward += 1.1 # +1 for destroy, +0.1 for deflect
                self._spawn_particles(block['rect'].center, block['color'])

                # Difficulty scaling
                if self.blocks_destroyed > 0 and self.blocks_destroyed % 20 == 0:
                    self.fall_speed += 0.05
                
                # Chance to spawn power-up
                if self.np_random.random() < self.POWERUP_CHANCE:
                    self._spawn_powerup(block['rect'].center)
            
            # Block hits bottom
            elif block['rect'].top > self.HEIGHT:
                # sfx: life_lost.wav
                self.blocks.remove(block)
                self.lives -= 1
                self.screen_flash_timer = 5 # Flash for 5 frames

        # Power-up collection
        for item in self.powerup_items[:]:
            if self.paddle.colliderect(item['rect']):
                # sfx: powerup_get.wav
                self.powerup_items.remove(item)
                reward += 5
                self.score += 50
                if item['type'] == 'extend_paddle':
                    self.active_powerups['extend_paddle'] = self.POWERUP_DURATION
                    center_x = self.paddle.centerx
                    self.paddle.width = self.PADDLE_WIDTH_EXTENDED
                    self.paddle.centerx = center_x
            
            # Remove off-screen powerups
            elif item['rect'].top > self.HEIGHT:
                self.powerup_items.remove(item)

        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })
            
    def _spawn_powerup(self, pos):
        self.powerup_items.append({
            'rect': pygame.Rect(pos[0] - self.POWERUP_SIZE // 2, pos[1], self.POWERUP_SIZE, self.POWERUP_SIZE),
            'type': 'extend_paddle' # Only one type for now
        })

    def _check_termination(self):
        terminated = False
        terminal_reward = 0

        if self.lives <= 0:
            terminated = True
            terminal_reward = -100
        elif self.blocks_destroyed >= self.TOTAL_BLOCKS:
            terminated = True
            terminal_reward = 100
            # sfx: win_jingle.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return terminated, terminal_reward

    def _get_observation(self):
        # --- Draw Background ---
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Render Game Elements ---
        self._render_game()
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            radius = int(p['lifespan'] * 0.2)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), max(0, radius), color
            )

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255,50), block['rect'].inflate(-6, -6), 1, border_radius=2)
            
        # Draw power-up items
        for item in self.powerup_items:
            hue = (self.steps * 5) % 360
            color = pygame.Color(0)
            color.hsva = (hue, 100, 100, 100)
            pygame.draw.rect(self.screen, color, item['rect'], border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, item['rect'], 2, border_radius=5)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw screen flash
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_FLASH)
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Blocks remaining
        blocks_text = self.font_small.render(f"BLOCKS LEFT: {max(0, self.TOTAL_BLOCKS - self.blocks_destroyed)}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.WIDTH // 2 - blocks_text.get_width() // 2, 10))

        # Game Over / Win message
        if self.game_over:
            if self.lives <= 0:
                msg = "GAME OVER"
            elif self.blocks_destroyed >= self.TOTAL_BLOCKS:
                msg = "YOU WIN!"
            else:
                msg = "TIME UP!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_PADDLE)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, text_rect.inflate(20, 20))
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, text_rect.inflate(20, 20), 2)
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_destroyed": self.blocks_destroyed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- To run and play the game ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    while running:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-closing or allow reset
            pygame.time.wait(2000)
            running = False

    env.close()