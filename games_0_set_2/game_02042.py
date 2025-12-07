
# Generated: 2025-08-27T19:04:20.140484
# Source Brief: brief_02042.md
# Brief Index: 2042

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Clear all the blocks without losing the ball."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.BALL_MAX_SPEED = 7
        self.BLOCK_COLS, self.BLOCK_ROWS = 10, 5
        self.NUM_BLOCKS = self.BLOCK_COLS * self.BLOCK_ROWS
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 18
        self.BLOCK_SPACING = 6
        self.INITIAL_LIVES = 3
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150, 60)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 50, 50)
        self.BLOCK_COLORS = [
            (100, 255, 100), (100, 100, 255), (255, 100, 100),
            (255, 255, 100), (100, 255, 255), (255, 100, 255)
        ]

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
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.chain_break_count = None
        self.last_paddle_hit_offset = None
        
        # This is here to ensure all attributes are defined before reset is called
        self.reset()

        # Validate implementation after full initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_launched = False
        self._reset_ball()
        
        self.blocks = self._generate_blocks()
        self.particles = []
        self.chain_break_count = 0
        self.last_paddle_hit_offset = 0

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_launched = False
    
    def _generate_blocks(self):
        blocks = []
        total_block_field_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_field_width) / 2
        start_y = 50
        
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(
                    start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING),
                    start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING),
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT
                )
                blocks.append({'rect': block_rect, 'color': color})
        return blocks

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        
        reward = -0.02  # Time penalty to encourage efficiency
        self.steps += 1
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle_rect.x))

        # Launch ball
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # Launch angle depends on position on paddle
            offset = (self.ball_pos[0] - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            angle = -math.pi / 2 - offset * (math.pi / 4) # Launch between -135 and -45 degrees
            
            self.ball_vel[0] = self.BALL_MAX_SPEED * math.cos(angle)
            self.ball_vel[1] = self.BALL_MAX_SPEED * math.sin(angle)
            self.chain_break_count = 0
            # // Sound effect: launch

        # 2. Update game state
        if not self.ball_launched:
            self.ball_pos[0] = self.paddle_rect.centerx
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

        self._handle_collisions()

        # 3. Update particles
        self._update_particles()
        
        # 4. Calculate reward (collision logic adds to this)
        reward += self.temp_reward
        self.temp_reward = 0 # Reset temporary reward

        # 5. Check termination conditions
        terminated = False
        if self.lives <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
        elif not self.blocks:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.temp_reward = 0

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # // Sound effect: wall_bounce

        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            self.ball_pos[1] = ball_rect.centery
            # // Sound effect: wall_bounce

        # Bottom collision (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
                # // Sound effect: lose_life
            else:
                self.game_over = True
                # // Sound effect: game_over
            if self.chain_break_count >= 5: # Bonus is lost if ball is dropped
                self.temp_reward -= 5
            self.chain_break_count = 0
            return # Skip other collisions for this frame

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            # // Sound effect: paddle_hit
            self.ball_vel[1] *= -1
            
            offset = (ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            # Clamp horizontal speed
            self.ball_vel[0] = max(-self.BALL_MAX_SPEED, min(self.BALL_MAX_SPEED, self.ball_vel[0]))

            # Ensure ball is above paddle after bounce
            ball_rect.bottom = self.paddle_rect.top
            self.ball_pos[1] = ball_rect.centery

            self.last_paddle_hit_offset = offset # Store for reward calculation
            
            if self.chain_break_count >= 5:
                self.temp_reward += 5
            self.chain_break_count = 0


        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # // Sound effect: block_break
                self._create_particles(block['rect'].center, block['color'])
                
                # Collision response
                prev_ball_rect = pygame.Rect(ball_rect)
                prev_ball_rect.x -= self.ball_vel[0]
                prev_ball_rect.y -= self.ball_vel[1]

                if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                    self.ball_vel[1] *= -1
                if prev_ball_rect.right <= block['rect'].left or prev_ball_rect.left >= block['rect'].right:
                    self.ball_vel[0] *= -1

                self.blocks.remove(block)
                self.score += 1
                self.temp_reward += 1
                self.chain_break_count += 1
                
                # Risky play reward based on last paddle hit
                if abs(self.last_paddle_hit_offset) > 0.7:
                    self.temp_reward += 0.1
                elif abs(self.last_paddle_hit_offset) < 0.3:
                    self.temp_reward -= 0.2

                break # Only hit one block per frame

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self._render_background()
        
        # Render all game elements
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_blocks(self):
        for block in self.blocks:
            r = block['rect']
            c = block['color']
            # Draw a slightly darker border
            pygame.draw.rect(self.screen, (max(0,c[0]-40),max(0,c[1]-40),max(0,c[2]-40)), r)
            # Draw the main block
            inner_rect = r.inflate(-4, -4)
            pygame.draw.rect(self.screen, c, inner_rect)


    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=4)
        # Add a subtle highlight
        highlight_rect = self.paddle_rect.copy()
        highlight_rect.height = 3
        pygame.draw.rect(self.screen, (255,255,255), highlight_rect, border_radius=2)


    def _render_ball(self):
        pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Main ball
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            self._draw_heart(self.WIDTH - 30 - i * 35, 25)
            
        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if not self.blocks else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(text_surf, text_rect)

    def _draw_heart(self, x, y):
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_HEART, points)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

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

# Example of how to run the environment
if __name__ == "__main__":
    # To run with a display, comment out the os.environ line at the top
    # and set render_mode="human" if that mode is implemented.
    # For this file, we will use a manual pygame loop to show the rgb_array.
    
    # Re-enable video driver for direct execution
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    pygame.display.init()
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    done = False
    
    print(env.user_guide)
    
    while not done:
        # Map pygame keys to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over. Final Info: {info}")
    env.close()