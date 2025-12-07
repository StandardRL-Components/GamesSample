
# Generated: 2025-08-27T22:02:33.607544
# Source Brief: brief_02992.md
# Brief Index: 2992

        
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
        "Controls: ←→ to move the paddle. Press space to launch the sonar pulse."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a submarine paddle to bounce a sonar pulse and shatter blocks of coral in this underwater Breakout-style game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000

    # Colors
    COLOR_BG_DARK = (10, 20, 40)
    COLOR_BG_LIGHT = (20, 40, 80)
    COLOR_PADDLE = (255, 200, 0)
    COLOR_PADDLE_LIGHT = (255, 255, 150)
    COLOR_PULSE = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_PARTICLE_RED = (255, 50, 50)

    # Paddle
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 20
    PADDLE_SPEED = 8.0

    # Pulse
    PULSE_RADIUS = 8
    PULSE_BASE_SPEED = 5.0
    PULSE_SPEED_INCREMENT = 0.05

    # Blocks
    BLOCK_ROWS = 5
    BLOCK_COLS = 10
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 6
    BLOCK_AREA_TOP = 50

    # Points and Colors for Blocks
    BLOCK_DATA = [
        {'points': 5, 'color': (255, 80, 80)},    # Red
        {'points': 5, 'color': (255, 120, 80)},   # Orange-Red
        {'points': 2, 'color': (80, 255, 80)},    # Green
        {'points': 2, 'color': (80, 200, 255)},   # Blue
        {'points': 1, 'color': (200, 80, 255)},   # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Etc...
        self.particles = []
        self.bg_particles = []
        self.kelp = []
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.chain_bonus = 0

        self.paddle_pos = pygame.Vector2(
            self.SCREEN_WIDTH / 2 - self.PADDLE_WIDTH / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        )

        self.pulse_in_play = False
        self.pulse_speed = self.PULSE_BASE_SPEED
        self._reset_pulse()

        self._create_blocks()
        self.blocks_destroyed = 0

        if not self.bg_particles:
            self._create_background_elements()

        self.particles.clear()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        self.game_over = self.steps >= self.MAX_STEPS

        self._handle_input(movement, space_held)
        
        if self.pulse_in_play:
            reward += 0.01  # Small reward for keeping pulse in play
            self._update_pulse()
            
            # Pulse-block collision
            hit_info = self._check_block_collisions()
            if hit_info:
                block = hit_info['block']
                
                # sfx_block_hit()
                self.score += block['points']
                self.score += self.chain_bonus
                reward += block['points'] + self.chain_bonus
                self.chain_bonus += 1
                
                block['alive'] = False
                self.blocks_destroyed += 1
                
                self._create_particles(self.pulse_pos, block['color'], 15)
                
                # Bounce
                if hit_info['side'] in ['top', 'bottom']:
                    self.pulse_vel.y *= -1
                else: # left, right
                    self.pulse_vel.x *= -1
                
                # Increase speed every 10 blocks
                if self.blocks_destroyed > 0 and self.blocks_destroyed % 10 == 0:
                    self.pulse_speed += self.PULSE_SPEED_INCREMENT
                    # sfx_speed_up()

        else: # Pulse is on the paddle
            self.pulse_pos.x = self.paddle_pos.x + self.PADDLE_WIDTH / 2
            self.pulse_pos.y = self.paddle_pos.y - self.PULSE_RADIUS

        # Check win/loss conditions
        if self.blocks_destroyed == len(self.blocks):
            self.win = True
            self.game_over = True
            reward += 100
            # sfx_win_game()
        
        if self.lives <= 0:
            self.game_over = True
            reward -= 100
            # sfx_game_over()

        self.steps += 1
        terminated = self.game_over
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos.x += self.PADDLE_SPEED
        
        self.paddle_pos.x = np.clip(self.paddle_pos.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Launch pulse
        if space_held and not self.pulse_in_play:
            self.pulse_in_play = True
            self.pulse_vel = pygame.Vector2(self.np_random.uniform(-1, 1), -1).normalize() * self.pulse_speed
            # sfx_launch_pulse()

    def _update_pulse(self):
        self.pulse_pos += self.pulse_vel

        # Wall collisions
        if self.pulse_pos.x <= self.PULSE_RADIUS or self.pulse_pos.x >= self.SCREEN_WIDTH - self.PULSE_RADIUS:
            self.pulse_vel.x *= -1
            self.pulse_pos.x = np.clip(self.pulse_pos.x, self.PULSE_RADIUS, self.SCREEN_WIDTH - self.PULSE_RADIUS)
            # sfx_wall_bounce()
        if self.pulse_pos.y <= self.PULSE_RADIUS:
            self.pulse_vel.y *= -1
            self.pulse_pos.y = np.clip(self.pulse_pos.y, self.PULSE_RADIUS, self.SCREEN_HEIGHT)
            # sfx_wall_bounce()

        # Lose life
        if self.pulse_pos.y >= self.SCREEN_HEIGHT:
            self.lives -= 1
            self.chain_bonus = 0
            self._create_particles(self.pulse_pos, self.COLOR_PARTICLE_RED, 30)
            self._reset_pulse()
            # sfx_lose_life()

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_pos.x, self.paddle_pos.y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if paddle_rect.colliderect(self._get_pulse_rect()):
            if self.pulse_vel.y > 0: # Only collide if moving downwards
                # sfx_paddle_bounce()
                self.chain_bonus = 0
                
                # Bounce logic
                self.pulse_pos.y = self.paddle_pos.y - self.PULSE_RADIUS
                
                offset = (self.pulse_pos.x - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
                bounce_angle = offset * (math.pi / 2.5) # Max angle ~72 degrees
                
                self.pulse_vel.x = self.pulse_speed * math.sin(bounce_angle)
                self.pulse_vel.y = -self.pulse_speed * math.cos(bounce_angle)
                self.pulse_vel.normalize_ip()
                self.pulse_vel *= self.pulse_speed
    
    def _check_block_collisions(self):
        pulse_rect = self._get_pulse_rect()
        for block in self.blocks:
            if block['alive'] and pulse_rect.colliderect(block['rect']):
                # Determine collision side to correctly reflect
                dx = (self.pulse_pos.x - block['rect'].centerx) / block['rect'].width
                dy = (self.pulse_pos.y - block['rect'].centery) / block['rect'].height
                
                side = 'none'
                if abs(dx) > abs(dy):
                    side = 'left' if dx < 0 else 'right'
                else:
                    side = 'top' if dy < 0 else 'bottom'
                    
                return {'block': block, 'side': side}
        return None

    def _reset_pulse(self):
        self.pulse_in_play = False
        self.pulse_pos = pygame.Vector2(
            self.paddle_pos.x + self.PADDLE_WIDTH / 2,
            self.paddle_pos.y - self.PULSE_RADIUS
        )
        self.pulse_vel = pygame.Vector2(0, 0)
    
    def _get_pulse_rect(self):
        return pygame.Rect(
            self.pulse_pos.x - self.PULSE_RADIUS,
            self.pulse_pos.y - self.PULSE_RADIUS,
            self.PULSE_RADIUS * 2,
            self.PULSE_RADIUS * 2
        )

    def _create_blocks(self):
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                block_data = self.BLOCK_DATA[r % len(self.BLOCK_DATA)]
                self.blocks.append({
                    'rect': pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    'color': block_data['color'],
                    'points': block_data['points'],
                    'alive': True
                })

    def _create_background_elements(self):
        # Plankton
        for _ in range(100):
            self.bg_particles.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'vel': pygame.Vector2(self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2)),
                'radius': self.np_random.uniform(0.5, 1.5),
                'color': (50, 80, 120, self.np_random.integers(50, 150))
            })
        # Kelp
        for _ in range(10):
            base_x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            kelp_strand = []
            for i in range(20):
                kelp_strand.append({
                    'offset': self.np_random.uniform(-10, 10),
                    'amp': self.np_random.uniform(5, 15)
                })
            self.kelp.append({'base_x': base_x, 'segments': kelp_strand})

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.uniform(20, 40),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Using a rect for the gradient is faster than line-by-line
        grad_rect = pygame.Surface((1, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_DARK[0] * (1 - interp) + self.COLOR_BG_LIGHT[0] * interp,
                self.COLOR_BG_DARK[1] * (1 - interp) + self.COLOR_BG_LIGHT[1] * interp,
                self.COLOR_BG_DARK[2] * (1 - interp) + self.COLOR_BG_LIGHT[2] * interp,
            )
            grad_rect.set_at((0, y), color)
        grad_rect = pygame.transform.scale(grad_rect, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.blit(grad_rect, (0, 0))

        # Update and render BG particles
        for p in self.bg_particles:
            p['pos'] += p['vel']
            p['pos'].x %= self.SCREEN_WIDTH
            p['pos'].y %= self.SCREEN_HEIGHT
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Render kelp
        for k in self.kelp:
            points = []
            for i, seg in enumerate(k['segments']):
                y = self.SCREEN_HEIGHT - i * 20
                x = k['base_x'] + seg['offset'] + math.sin(self.steps * 0.02 + i * 0.3) * seg['amp']
                points.append((x,y))
            if len(points) > 1:
                pygame.draw.aalines(self.screen, (20, 100, 60), False, points, 2)

    def _render_game_elements(self):
        # Render blocks
        for block in self.blocks:
            if block['alive']:
                pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
                # Inner highlight
                highlight_rect = block['rect'].inflate(-6, -6)
                highlight_color = tuple(min(255, c + 40) for c in block['color'])
                pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=3)

        # Render paddle (submarine)
        paddle_rect = pygame.Rect(self.paddle_pos.x, self.paddle_pos.y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=5)
        # Cockpit
        cockpit_pos = (int(paddle_rect.centerx), int(paddle_rect.centery - 2))
        pygame.gfxdraw.filled_circle(self.screen, cockpit_pos[0], cockpit_pos[1], 5, self.COLOR_PADDLE_LIGHT)
        pygame.gfxdraw.aacircle(self.screen, cockpit_pos[0], cockpit_pos[1], 5, self.COLOR_PADDLE)

        # Render pulse
        if self.pulse_in_play or self.lives > 0:
            # Sonar ping effect
            ping_radius = (self.steps % 30) * 1.5
            ping_alpha = max(0, 255 - ping_radius * 5)
            if ping_alpha > 0:
                color = self.COLOR_PULSE + (ping_alpha,)
                pygame.gfxdraw.aacircle(self.screen, int(self.pulse_pos.x), int(self.pulse_pos.y), int(ping_radius), color)
            
            # Pulse core
            pygame.gfxdraw.filled_circle(self.screen, int(self.pulse_pos.x), int(self.pulse_pos.y), self.PULSE_RADIUS, self.COLOR_PULSE)
            pygame.gfxdraw.aacircle(self.screen, int(self.pulse_pos.x), int(self.pulse_pos.y), self.PULSE_RADIUS, (255,255,255))
            
    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.98 # friction
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 40))
                r,g,b = p['color']
                color_with_alpha = (r,g,b,alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), max(0, int(p['radius'])), color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            pos_x = self.SCREEN_WIDTH - 30 - i * (self.PULSE_RADIUS * 2 + 10)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 25, self.PULSE_RADIUS, self.COLOR_PULSE)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 25, self.PULSE_RADIUS, (255,255,255))

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Coral Breakout")
    clock = pygame.time.Clock()

    while running:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {round(total_reward, 2)}")
            # Render final frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            # A small delay before restarting
            pygame.time.wait(2000)
            
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Control frame rate
        clock.tick(30)
        
    env.close()