
# Generated: 2025-08-27T23:44:01.524553
# Source Brief: brief_03564.md
# Brief Index: 3564

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Break all the blocks in this fast-paced arcade block breaker with procedurally generated levels and strategic power-ups."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_BOUNDARY = (100, 100, 120)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 0, 50)
    
    # Block Colors by Health
    BLOCK_COLORS = {
        1: (0, 150, 255),  # Blue
        2: (0, 255, 150),  # Green
        3: (255, 100, 0),  # Red
    }
    
    # Power-up Colors
    POWERUP_COLORS = {
        "EXTRA_BALL": (255, 255, 0),    # Yellow
        "PADDLE_SIZE": (200, 0, 255), # Purple
    }

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400
    
    # Game Parameters
    PADDLE_SPEED = 8
    PADDLE_WIDTH_INITIAL = 100
    PADDLE_HEIGHT = 15
    BALL_RADIUS = 7
    BALL_SPEED_INITIAL = 5
    BALL_SPEED_MAX = 8
    MAX_LIVES = 3
    TOTAL_BLOCKS = 100
    MAX_STEPS = 1000 * 5 # ~5 minutes at 30fps
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        # Initialize state variables
        self.paddle = None
        self.balls = []
        self.blocks = []
        self.powerups = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.game_won = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.game_won = False

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH_INITIAL) / 2, 
            self.HEIGHT - self.PADDLE_HEIGHT - 10, 
            self.PADDLE_WIDTH_INITIAL, 
            self.PADDLE_HEIGHT
        )
        
        self.balls = []
        self._create_ball(attached=True)

        self.blocks = self._generate_blocks()
        self.powerups = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02 # Small penalty for each step to encourage action
        block_hit_this_step = False
        
        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1

        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(1, self.paddle.left)
        self.paddle.right = min(self.WIDTH - 1, self.paddle.right)

        if space_held:
            for ball in self.balls:
                if ball['attached']:
                    ball['attached'] = False
                    # Sound: Ball Launch
                    break

        # --- Update Game Logic ---
        self._update_balls()
        
        # Ball-Block Collision
        for ball in self.balls:
            if ball['attached']: continue
            
            collided_block_idx = ball['rect'].collidelist([b['rect'] for b in self.blocks])
            if collided_block_idx != -1:
                block = self.blocks[collided_block_idx]
                
                # Determine bounce direction
                self._handle_ball_block_bounce(ball, block)
                
                # Process block hit
                block['health'] -= 1
                reward += 0.1 # Reward for hitting a block
                block_hit_this_step = True
                # Sound: Block Hit
                
                if block['health'] <= 0:
                    reward += 1 # Reward for destroying a block
                    self._create_particles(block['rect'].center, block['color'])
                    # Sound: Block Destroy
                    
                    # Chance to drop a power-up
                    if self.np_random.random() < 0.15: # 15% chance
                        self._create_powerup(block['rect'].center)
                    
                    self.blocks.pop(collided_block_idx)
        
        if not block_hit_this_step:
            reward = -0.02 # Reset to penalty if no block was hit
        else:
            reward = max(0, reward) # Ensure reward is not negative if a block was hit

        self._update_powerups()
        reward += self._check_powerup_collection()
        self._update_particles()
        
        # --- Check Termination ---
        self.steps += 1
        terminated = False
        
        if not self.blocks:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100 # Win bonus
        elif self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Lose penalty
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
        }

    # --- Helper Methods: Game Logic ---
    
    def _create_ball(self, attached=False):
        ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if attached:
            ball_rect.center = self.paddle.centerx, self.paddle.top - self.BALL_RADIUS
        else:
            ball_rect.center = self.paddle.center
        
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards
        
        self.balls.append({
            'rect': ball_rect,
            'vel': [math.cos(angle) * self.BALL_SPEED_INITIAL, math.sin(angle) * self.BALL_SPEED_INITIAL],
            'attached': attached,
        })

    def _update_balls(self):
        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            if ball['attached']:
                ball['rect'].centerx = self.paddle.centerx
                ball['rect'].bottom = self.paddle.top
                continue

            ball['rect'].x += ball['vel'][0]
            ball['rect'].y += ball['vel'][1]

            # Wall collisions
            if ball['rect'].left <= 0 or ball['rect'].right >= self.WIDTH:
                ball['vel'][0] *= -1
                ball['rect'].left = max(0, ball['rect'].left)
                ball['rect'].right = min(self.WIDTH, ball['rect'].right)
                # Sound: Wall Bounce
            if ball['rect'].top <= 0:
                ball['vel'][1] *= -1
                ball['rect'].top = max(0, ball['rect'].top)
                # Sound: Wall Bounce

            # Paddle collision
            if ball['rect'].colliderect(self.paddle) and ball['vel'][1] > 0:
                ball['vel'][1] *= -1
                
                # Influence horizontal velocity based on hit location
                offset = (ball['rect'].centerx - self.paddle.centerx) / (self.paddle.width / 2)
                ball['vel'][0] += offset * 2.0
                
                # Clamp ball speed
                speed = math.hypot(*ball['vel'])
                if speed > self.BALL_SPEED_MAX:
                    scale = self.BALL_SPEED_MAX / speed
                    ball['vel'][0] *= scale
                    ball['vel'][1] *= scale
                # Sound: Paddle Bounce

            # Out of bounds (lost ball)
            if ball['rect'].top > self.HEIGHT:
                balls_to_remove.append(i)
        
        # Remove lost balls
        for i in sorted(balls_to_remove, reverse=True):
            self.balls.pop(i)

        # Handle life loss
        if not self.balls and not self.game_over:
            self.lives -= 1
            if self.lives > 0:
                self._create_ball(attached=True)
                # Sound: Life Lost

    def _handle_ball_block_bounce(self, ball, block):
        # A simple but effective bounce logic for AABB
        ball_rect = ball['rect']
        block_rect = block['rect']
        
        overlap = ball_rect.clip(block_rect)
        
        if overlap.width < overlap.height:
            # Horizontal collision
            ball['vel'][0] *= -1
            # Nudge ball out of collision
            if ball_rect.centerx < block_rect.centerx:
                ball_rect.right = block_rect.left
            else:
                ball_rect.left = block_rect.right
        else:
            # Vertical collision
            ball['vel'][1] *= -1
            # Nudge ball out of collision
            if ball_rect.centery < block_rect.centery:
                ball_rect.bottom = block_rect.top
            else:
                ball_rect.top = block_rect.bottom

    def _generate_blocks(self):
        blocks = []
        block_width, block_height = 40, 20
        gap = 4
        cols = self.WIDTH // (block_width + gap)
        rows = 12
        
        attempts = 0
        while len(blocks) < self.TOTAL_BLOCKS and attempts < 1000:
            attempts += 1
            r = self.np_random.integers(0, rows)
            c = self.np_random.integers(0, cols)
            
            # Lower rows have lower probability to ensure starting gaps
            if self.np_random.random() > (r / rows) * 0.5 + 0.5:
                continue

            x = c * (block_width + gap) + gap * 2
            y = r * (block_height + gap) + 40
            
            rect = pygame.Rect(x, y, block_width, block_height)
            
            # Avoid duplicates
            if any(b['rect'] == rect for b in blocks):
                continue
            
            health = self.np_random.integers(1, 4)
            blocks.append({'rect': rect, 'health': health, 'color': self.BLOCK_COLORS[health]})
        
        return blocks

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(10, 25)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _create_powerup(self, pos):
        powerup_type = self.np_random.choice(["EXTRA_BALL", "PADDLE_SIZE"])
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = pos
        self.powerups.append({'rect': rect, 'type': powerup_type, 'vel_y': 2})

    def _update_powerups(self):
        self.powerups = [p for p in self.powerups if p['rect'].top < self.HEIGHT]
        for p in self.powerups:
            p['rect'].y += p['vel_y']
            
    def _check_powerup_collection(self):
        collected_reward = 0
        powerups_to_remove = []
        for i, p in enumerate(self.powerups):
            if self.paddle.colliderect(p['rect']):
                self._apply_powerup(p['type'])
                collected_reward += 5
                powerups_to_remove.append(i)
                # Sound: Power-up Collect
        
        for i in sorted(powerups_to_remove, reverse=True):
            self.powerups.pop(i)
        
        return collected_reward

    def _apply_powerup(self, type):
        self.score += 50 # Bonus score for powerup
        if type == "EXTRA_BALL":
            self._create_ball(attached=False)
        elif type == "PADDLE_SIZE":
            self.paddle.inflate_ip(40, 0)
            self.paddle.width = min(self.paddle.width, self.WIDTH // 2)

    # --- Helper Methods: Rendering ---

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Draw play area boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS.get(block['health'], (255, 255, 255))
            pygame.draw.rect(self.screen, color, block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Draw power-ups
        for p in self.powerups:
            color = self.POWERUP_COLORS[p['type']]
            pygame.gfxdraw.box(self.screen, p['rect'], color)
            pygame.gfxdraw.aacircle(self.screen, p['rect'].centerx, p['rect'].centery, p['rect'].width//2 - 2, (255,255,255))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(self.BALL_RADIUS / 2 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw balls
        for ball in self.balls:
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, ball['rect'].centerx, ball['rect'].centery, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
            # Main ball
            pygame.gfxdraw.aacircle(self.screen, ball['rect'].centerx, ball['rect'].centery, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, ball['rect'].centerx, ball['rect'].centery, self.BALL_RADIUS, self.COLOR_BALL)
            
    def _render_ui(self):
        # Draw score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, (200, 200, 255))
        self.screen.blit(score_surf, (10, 10))

        # Draw lives
        for i in range(self.lives):
            x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw game over/win message
        if self.game_over:
            msg_text = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 255, 128) if self.game_won else (255, 50, 50)
            msg_surf = self.font_msg.render(msg_text, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    # To play, comment out the line above. To run headless, keep it.

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")
    
    while running:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused in this game

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Game Loop Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            print("Press 'R' to play again, or close the window to quit.")
            # Wait for reset
            while running:
                reset_pressed = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        reset_pressed = True
                        break
                if not running or reset_pressed:
                    break

        clock.tick(60) # Run at 60 FPS for smooth human play

    env.close()