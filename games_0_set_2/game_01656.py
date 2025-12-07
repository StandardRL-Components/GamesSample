
# Generated: 2025-08-27T17:51:51.750265
# Source Brief: brief_01656.md
# Brief Index: 1656

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, visually vibrant block breaker where risky plays are rewarded and hesitation is penalized. Clear all the blocks to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WALL_THICKNESS = 10

        # Colors
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (100, 100, 255)
        self.COLOR_BALL = (50, 255, 50)
        self.COLOR_BALL_GLOW = (50, 255, 50)
        self.COLOR_WALL = (0, 200, 255)
        self.COLOR_POWERUP = (255, 223, 0)
        self.COLOR_TEXT = (255, 255, 0)
        self.BLOCK_COLORS = [
            (255, 0, 0), (255, 127, 0), (255, 255, 0), 
            (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)
        ]

        # Game constants
        self.MAX_STEPS = 2500
        self.INITIAL_BALLS = 5
        self.PADDLE_Y = self.HEIGHT - 30
        self.PADDLE_HEIGHT = 10
        self.PADDLE_BASE_WIDTH = 100
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 6.0
        self.POWERUP_CHANCE = 0.15
        self.POWERUP_SPEED = 2.0
        self.POWERUP_SIZE = 12
        self.POWERUP_DURATION = 300 # 10 seconds at 30fps

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.paddle_x = 0
        self.paddle_width = 0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.active_powerups = {}
        self.balls_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.paddle_x = self.WIDTH / 2
        self.ball_attached = True
        self.particles.clear()
        self.powerups.clear()
        self.active_powerups.clear()

        self._reset_ball()
        self._create_block_layout()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        self.steps += 1
        reward -= 0.02 # Penalty for time passing

        self._handle_input(movement, space_held)
        self._update_powerups()
        
        if not self.ball_attached:
            ball_reward = self._move_ball()
            reward += ball_reward

        self._update_particles()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated
        
        self.score += reward # Score is just cumulative reward for this game
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3: # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_x += self.PADDLE_SPEED
        
        self.paddle_width = self.PADDLE_BASE_WIDTH * (1.5 if self._is_powerup_active('wide_paddle') else 1.0)
        self.paddle_x = np.clip(self.paddle_x, self.WALL_THICKNESS + self.paddle_width / 2, self.WIDTH - self.WALL_THICKNESS - self.paddle_width / 2)

        # Launch ball
        if space_held and self.ball_attached:
            self.ball_attached = False
            # Launch with a slight random angle
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_SPEED
            # Sound effect placeholder: // Launch sound

    def _move_ball(self):
        reward = 0.0
        self.ball_pos += self.ball_vel
        
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.left = self.WALL_THICKNESS
        if ball_rect.right >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.right = self.WIDTH - self.WALL_THICKNESS
        if ball_rect.top <= self.WALL_THICKNESS:
            self.ball_vel.y *= -1
            ball_rect.top = self.WALL_THICKNESS
        
        self.ball_pos.x = ball_rect.centerx
        self.ball_pos.y = ball_rect.centery

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x - self.paddle_width / 2, self.PADDLE_Y, self.paddle_width, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vel.y > 0:
            # Sound effect placeholder: // Paddle hit sound
            hit_pos = (self.ball_pos.x - self.paddle_x) / (self.paddle_width / 2)
            hit_pos = np.clip(hit_pos, -1, 1)
            
            angle = hit_pos * (math.pi / 2.5) # Max angle ~72 degrees
            
            new_vx = self.BALL_SPEED * math.sin(angle)
            new_vy = -self.BALL_SPEED * math.cos(angle)
            self.ball_vel = pygame.Vector2(new_vx, new_vy)
            
            # Ensure ball is above paddle to prevent getting stuck
            self.ball_pos.y = self.PADDLE_Y - self.BALL_RADIUS

            # Risky play reward
            if abs(angle) > math.pi / 4: # 45 degrees
                reward += 0.1

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # Sound effect placeholder: // Block break sound
                reward += 1.0
                
                # Determine bounce direction
                prev_ball_pos = self.ball_pos - self.ball_vel
                if (prev_ball_pos.x < block['rect'].left or prev_ball_pos.x > block['rect'].right):
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1

                self._spawn_particles(block['rect'].center, block['color'])
                if self.np_random.random() < self.POWERUP_CHANCE:
                    self._spawn_powerup(block['rect'].center)
                
                self.blocks.remove(block)
                break # Only one block per frame

        # Ball lost
        if self.ball_pos.y > self.HEIGHT:
            self.balls_left -= 1
            self.ball_attached = True
            self._reset_ball()
            if self.balls_left > 0:
                # Sound effect placeholder: // Lose life sound
                self._create_screen_flash((255, 50, 50), 15)
            else:
                # Sound effect placeholder: // Game over sound
                pass

        return reward

    def _update_powerups(self):
        # Decrement active powerup timers
        for key in list(self.active_powerups.keys()):
            self.active_powerups[key] -= 1
            if self.active_powerups[key] <= 0:
                del self.active_powerups[key]
        
        # Move falling powerups
        paddle_rect = pygame.Rect(self.paddle_x - self.paddle_width / 2, self.PADDLE_Y, self.paddle_width, self.PADDLE_HEIGHT)
        for p in self.powerups[:]:
            p['pos'].y += self.POWERUP_SPEED
            p_rect = pygame.Rect(p['pos'].x - self.POWERUP_SIZE / 2, p['pos'].y - self.POWERUP_SIZE / 2, self.POWERUP_SIZE, self.POWERUP_SIZE)
            
            if p_rect.colliderect(paddle_rect):
                # Sound effect placeholder: // Power-up collect sound
                self.score += 5.0
                self.active_powerups[p['type']] = self.POWERUP_DURATION
                self.powerups.remove(p)
                self._create_screen_flash(self.COLOR_POWERUP, 10)
            elif p['pos'].y > self.HEIGHT:
                self.powerups.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.balls_left <= 0:
            return True, -100.0
        if not self.blocks:
            return True, 100.0
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
        return False, 0.0

    def _get_observation(self):
        self._render_background()
        self._render_walls()
        self._render_blocks()
        self._render_powerups()
        self._render_particles()
        self._render_paddle()
        self._render_ball()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

    def _render_paddle(self):
        paddle_rect = pygame.Rect(0, 0, self.paddle_width, self.PADDLE_HEIGHT)
        paddle_rect.center = (int(self.paddle_x), int(self.PADDLE_Y + self.PADDLE_HEIGHT / 2))
        
        # Glow effect
        glow_rect = paddle_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE_GLOW, 60), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=5)

    def _render_ball(self):
        if self.ball_attached:
            self.ball_pos.x = self.paddle_x
            self.ball_pos.y = self.PADDLE_Y - self.BALL_RADIUS

        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), glow_radius, (*self.COLOR_BALL_GLOW, 30))
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), int(glow_radius * 0.6), (*self.COLOR_BALL_GLOW, 50))
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, (255,255,255), block['rect'], 1) # White outline

    def _render_powerups(self):
        for p in self.powerups:
            # Flashing effect
            alpha = 128 + 127 * math.sin(self.steps * 0.3)
            color = (*self.COLOR_POWERUP, alpha)
            
            rect = pygame.Rect(p['pos'].x - self.POWERUP_SIZE/2, p['pos'].y - self.POWERUP_SIZE/2, self.POWERUP_SIZE, self.POWERUP_SIZE)
            temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect(), border_radius=3)
            self.screen.blit(temp_surf, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Use a surface for transparency
                part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(part_surf, color, (size, size), size)
                self.screen.blit(part_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))

        # Balls left
        for i in range(self.balls_left - (1 if self.ball_attached else 0)):
            pos_x = self.WIDTH - self.WALL_THICKNESS - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.draw.circle(self.screen, self.COLOR_PADDLE, (pos_x, self.WALL_THICKNESS + 20), self.BALL_RADIUS)
        
        # Game Over message
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
                color = (50, 255, 50)
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_block_layout(self):
        self.blocks.clear()
        block_width = 50
        block_height = 20
        num_cols = 11
        num_rows = 7
        gap = 4
        total_block_width = num_cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 60

        for r in range(num_rows):
            for c in range(num_cols):
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                rect = pygame.Rect(x, y, block_width, block_height)
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                self.blocks.append({'rect': rect, 'color': color})

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle_x, self.PADDLE_Y - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
    
    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 40)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _spawn_powerup(self, pos):
        self.powerups.append({
            'pos': pygame.Vector2(pos),
            'type': 'wide_paddle' # Only one type for now
        })

    def _is_powerup_active(self, powerup_type):
        return powerup_type in self.active_powerups and self.active_powerups[powerup_type] > 0

    def _create_screen_flash(self, color, duration):
        # This is a visual effect that can't be easily implemented without a main loop.
        # In a gym env, we just render the state. A flash would need to be a state variable.
        # For simplicity, this is omitted, but a real game would use it.
        pass

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

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to our action space
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    print("--- Controls ---")
    print(env.user_guide)
    print("----------------")

    # Main game loop
    running = True
    while running:
        # Pygame rendering for human play
        render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            done = False

        # Action defaults
        movement = 0 # no-op
        space = 0 # released
        shift = 0 # released

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Control FPS
        env.clock.tick(30)

    env.close()