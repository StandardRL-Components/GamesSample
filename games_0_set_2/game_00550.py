
# Generated: 2025-08-27T13:59:11.638910
# Source Brief: brief_00550.md
# Brief Index: 550

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball. Press shift to activate a power-up."
    )

    game_description = (
        "A fast-paced, procedurally generated block-breaking game where strategic paddle positioning and power-up utilization are key to achieving high scores."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 100)
        self.COLOR_BALL_GLOW = (200, 200, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.BLOCK_COLORS = [
            (255, 80, 80), (255, 160, 80), (255, 255, 80),
            (80, 255, 80), (80, 160, 255), (160, 80, 255)
        ]

        # Game constants
        self.PADDLE_BASE_WIDTH = 80
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 5
        self.MAX_LIVES = 3
        self.MAX_STEPS = 10000
        self.POWERUP_CHANCE = 0.25
        self.POWERUP_DURATION = 600 # 20 seconds at 30fps
        self.POWERUP_FALL_SPEED = 2

        # Initialize state variables
        self.paddle_rect = None
        self.balls = []
        self.blocks = []
        self.particles = []
        self.falling_powerups = []
        self.stored_powerup = None
        self.active_powerups = {}
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.steps_since_last_hit = 0
        self.ball_speed_multiplier = 1.0
        self.shift_cooldown = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_BASE_WIDTH) // 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_BASE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.balls = []
        self._reset_ball()
        
        self.particles = []
        self.falling_powerups = []
        self.stored_powerup = None
        self.active_powerups = {}
        
        self.steps_since_last_hit = 0
        self.ball_speed_multiplier = 1.0
        self.shift_cooldown = 0

        self._generate_blocks()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for every step
        
        self._handle_input(action)
        self._update_game_state()
        
        hit_block_this_step = self._handle_collisions()
        if hit_block_this_step:
            reward += 0.1
            self.steps_since_last_hit = 0
        else:
            self.steps_since_last_hit += 1

        # Anti-softlock
        if self.steps_since_last_hit > 500 and self.blocks:
            destroyed_block = self.blocks.pop(self.np_random.integers(len(self.blocks)))
            self._spawn_particles(destroyed_block.center, random.choice(self.BLOCK_COLORS))
            self.steps_since_last_hit = 0
            # sfx: block_break_softlock.wav

        reward += self._update_balls()

        terminated = self._check_termination()
        
        if terminated:
            if not self.blocks: # Win condition
                reward = 100
            else: # Lose condition
                reward = -100
        
        # Add reward for clearing the last block
        if hit_block_this_step and not self.blocks:
            reward += 5
        
        # Reward for activating a power-up
        if self.just_activated_powerup:
            reward += 1
            self.just_activated_powerup = False
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Paddle movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        self.paddle_rect.left = max(0, self.paddle_rect.left)
        self.paddle_rect.right = min(self.SCREEN_WIDTH, self.paddle_rect.right)

        # Launch ball
        if space_held:
            for ball in self.balls:
                if not ball['launched']:
                    ball['launched'] = True
                    initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                    speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
                    ball['vel'] = [math.cos(initial_angle) * speed, math.sin(initial_angle) * speed]
                    # sfx: ball_launch.wav

        # Activate power-up
        self.just_activated_powerup = False
        if shift_held and self.stored_powerup and self.shift_cooldown == 0:
            self._activate_powerup()
            self.just_activated_powerup = True
            self.shift_cooldown = 10 # 1/3 second cooldown

    def _update_game_state(self):
        # Update shift cooldown
        if self.shift_cooldown > 0:
            self.shift_cooldown -= 1

        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
            self.ball_speed_multiplier += 0.05

        # Update active powerups
        for key in list(self.active_powerups.keys()):
            self.active_powerups[key] -= 1
            if self.active_powerups[key] <= 0:
                del self.active_powerups[key]
                if key == 'large_paddle':
                    self.paddle_rect.width = self.PADDLE_BASE_WIDTH
                    self.paddle_rect.inflate_ip(0, 0) # Recenter

        # Apply large paddle effect
        if 'large_paddle' in self.active_powerups:
            self.paddle_rect.width = self.PADDLE_BASE_WIDTH * 1.5
        else:
            self.paddle_rect.width = self.PADDLE_BASE_WIDTH

        # Update falling powerups
        for p_up in self.falling_powerups[:]:
            p_up['rect'].y += self.POWERUP_FALL_SPEED
            if p_up['rect'].colliderect(self.paddle_rect):
                if not self.stored_powerup: # Only store one
                    self.stored_powerup = p_up['type']
                    # sfx: powerup_collect.wav
                self.falling_powerups.remove(p_up)
            elif p_up['rect'].top > self.SCREEN_HEIGHT:
                self.falling_powerups.remove(p_up)

        # Update particles
        self.particles = [p for p in self.particles if self._update_particle(p)]

    def _update_particle(self, p):
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        p['lifespan'] -= 1
        return p['lifespan'] > 0

    def _handle_collisions(self):
        hit_block_this_step = False
        for ball in self.balls:
            if not ball['launched']:
                continue

            ball['rect'].x += ball['vel'][0]
            ball['rect'].y += ball['vel'][1]

            # Wall collisions
            if ball['rect'].left < 0 or ball['rect'].right > self.SCREEN_WIDTH:
                ball['vel'][0] *= -1
                ball['rect'].left = max(0, ball['rect'].left)
                ball['rect'].right = min(self.SCREEN_WIDTH, ball['rect'].right)
                # sfx: wall_bounce.wav
            if ball['rect'].top < 0:
                ball['vel'][1] *= -1
                ball['rect'].top = max(0, ball['rect'].top)
                # sfx: wall_bounce.wav

            # Paddle collision
            if ball['rect'].colliderect(self.paddle_rect) and ball['vel'][1] > 0:
                ball['vel'][1] *= -1
                
                # Influence horizontal velocity based on hit position
                offset = (ball['rect'].centerx - self.paddle_rect.centerx) / (self.paddle_rect.width / 2)
                ball['vel'][0] += offset * 2.0
                
                # Normalize speed
                speed = math.sqrt(ball['vel'][0]**2 + ball['vel'][1]**2)
                target_speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
                if speed > 0:
                    ball['vel'][0] = (ball['vel'][0] / speed) * target_speed
                    ball['vel'][1] = (ball['vel'][1] / speed) * target_speed
                
                ball['rect'].bottom = self.paddle_rect.top
                # sfx: paddle_bounce.wav
                
            # Block collisions
            for block in self.blocks[:]:
                if ball['rect'].colliderect(block['rect']):
                    hit_block_this_step = True
                    self.score += 10
                    self._spawn_particles(block['rect'].center, block['color'])
                    self.blocks.remove(block)
                    # sfx: block_break.wav

                    # Spawn powerup
                    if self.np_random.random() < self.POWERUP_CHANCE:
                        self._spawn_powerup(block['rect'].center)

                    # Bounce logic
                    prev_rect = ball['rect'].copy()
                    prev_rect.x -= ball['vel'][0]
                    prev_rect.y -= ball['vel'][1]

                    if prev_rect.bottom <= block['rect'].top or prev_rect.top >= block['rect'].bottom:
                        ball['vel'][1] *= -1
                    if prev_rect.right <= block['rect'].left or prev_rect.left >= block['rect'].right:
                        ball['vel'][0] *= -1
                    break # Only hit one block per frame per ball
        return hit_block_this_step

    def _update_balls(self):
        balls_to_remove = []
        for ball in self.balls:
            if ball['rect'].top > self.SCREEN_HEIGHT:
                balls_to_remove.append(ball)
        
        for ball in balls_to_remove:
            self.balls.remove(ball)

        if not self.balls and not self.game_over:
            self.lives -= 1
            # sfx: lose_life.wav
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
        return 0

    def _reset_ball(self):
        ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        ball_rect.center = self.paddle_rect.center
        ball_rect.bottom = self.paddle_rect.top
        self.balls.append({
            'rect': ball_rect,
            'vel': [0, 0],
            'launched': False
        })
    
    def _activate_powerup(self):
        if self.stored_powerup == 'large_paddle':
            self.active_powerups['large_paddle'] = self.POWERUP_DURATION
            # sfx: powerup_activate_large.wav
        elif self.stored_powerup == 'multi_ball' and self.balls:
            primary_ball = self.balls[0]
            for i in range(2):
                new_ball_rect = primary_ball['rect'].copy()
                angle = self.np_random.uniform(-math.pi, math.pi)
                speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
                new_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                
                # Ensure new balls have upward velocity if possible
                if new_vel[1] > 0 and primary_ball['vel'][1] < 0:
                    new_vel[1] *= -1
                
                self.balls.append({
                    'rect': new_ball_rect,
                    'vel': new_vel,
                    'launched': True
                })
            # sfx: powerup_activate_multi.wav
        self.stored_powerup = None

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.blocks:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _generate_blocks(self):
        self.blocks = []
        block_width = 40
        block_height = 20
        num_cols = self.SCREEN_WIDTH // block_width
        num_rows = 6

        for r in range(num_rows):
            for c in range(num_cols):
                # Procedural generation: leave some gaps
                if self.np_random.random() > 0.15:
                    block_rect = pygame.Rect(
                        c * block_width,
                        r * block_height + 50,
                        block_width,
                        block_height
                    )
                    self.blocks.append({'rect': block_rect, 'color': self.BLOCK_COLORS[r]})

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _spawn_powerup(self, pos):
        powerup_rect = pygame.Rect(0, 0, 20, 20)
        powerup_rect.center = pos
        powerup_type = self.np_random.choice(['large_paddle', 'multi_ball'])
        self.falling_powerups.append({'rect': powerup_rect, 'type': powerup_type})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Balls
        for ball in self.balls:
            if not ball['launched']: # Attach to paddle if not launched
                ball['rect'].centerx = self.paddle_rect.centerx
                ball['rect'].bottom = self.paddle_rect.top
            
            # Glow effect
            glow_radius = int(self.BALL_RADIUS * 1.8)
            glow_center = ball['rect'].center
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_BALL_GLOW, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (glow_center[0] - glow_radius, glow_center[1] - glow_radius))
            
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, int(ball['rect'].centerx), int(ball['rect'].centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            color = (*p['color'], alpha)
            size = int(max(1, self.BALL_RADIUS / 3 * (p['lifespan'] / 30)))
            pygame.draw.circle(self.screen, color, [int(c) for c in p['pos']], size)

        # Falling Powerups
        for p_up in self.falling_powerups:
            # Flashing rainbow effect
            hue = (self.steps * 10) % 360
            color = pygame.Color(0)
            color.hsva = (hue, 100, 100, 100)
            pygame.draw.rect(self.screen, color, p_up['rect'], border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, p_up['rect'], 2, border_radius=5)
            
            # Icon
            p_char = 'P' if p_up['type'] == 'large_paddle' else 'M'
            text = self.font_small.render(p_char, True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=p_up['rect'].center))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Lives
        life_icon = pygame.Surface((self.PADDLE_BASE_WIDTH / 3, self.PADDLE_HEIGHT / 2))
        life_icon.fill(self.COLOR_PADDLE)
        for i in range(self.lives):
            self.screen.blit(life_icon, (self.SCREEN_WIDTH - 20 - (i + 1) * (life_icon.get_width() + 5), 15))

        # Stored Powerup
        if self.stored_powerup:
            powerup_box = pygame.Rect(10, 45, 100, 25)
            pygame.draw.rect(self.screen, self.COLOR_GRID, powerup_box, border_radius=5)
            
            text_str = self.stored_powerup.replace('_', ' ').title()
            text = self.font_small.render(text_str, True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=powerup_box.center))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            text = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "stored_powerup": self.stored_powerup
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
        
        # Test specific game mechanics
        self.reset()
        initial_speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
        self.steps = 49
        self.step(self.action_space.sample()) # step 50
        assert self.ball_speed_multiplier > 1.0
        
        self.reset()
        initial_score = self.score
        self.blocks = [{'rect': pygame.Rect(300, 300, 20, 20), 'color': (255,0,0)}]
        self.balls = [{'rect': pygame.Rect(300, 290, 14, 14), 'vel': [0, 5], 'launched': True}]
        self.step(self.action_space.sample())
        assert self.score > initial_score
        assert not self.blocks

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    # Remap keys for human play
    key_map = {
        pygame.K_LEFT: (0, 3),   # action index 0, value 3
        pygame.K_RIGHT: (0, 4),  # action index 0, value 4
        pygame.K_SPACE: (1, 1),  # action index 1, value 1
        pygame.K_LSHIFT: (2, 1), # action index 2, value 1
        pygame.K_RSHIFT: (2, 1)
    }

    # Pygame setup for rendering
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map pressed keys to actions
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            
        # Control frame rate
        env.clock.tick(30)

    env.close()