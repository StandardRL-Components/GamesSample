
# Generated: 2025-08-28T05:51:12.510153
# Source Brief: brief_05712.md
# Brief Index: 5712

        
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
        "Controls: ←→ to aim, Space to launch the ball. Clear all the blocks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, geometric block-breaking game. Aim your shots to clear the "
        "grid, using bounces and chain reactions to your advantage. You have 3 balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_LAUNCHER = (100, 100, 120)
        self.COLOR_AIM_LINE = (255, 255, 255)
        self.BLOCK_COLORS = [
            (66, 135, 245),  # Health 1
            (55, 110, 220),  # Health 2
            (45, 90, 190),   # Health 3
            (35, 70, 160),   # Health 4
            (25, 50, 130),   # Health 5
        ]
        self.PARTICLE_COLORS = [(255, 80, 80), (255, 150, 80), (255, 255, 100)]

        # Game parameters
        self.BALL_RADIUS = 5
        self.BALL_SPEED = 10
        self.BLOCK_ROWS = 10
        self.BLOCK_COLS = 10
        self.GRID_TOP_MARGIN = 50
        self.GRID_SIDE_MARGIN = 40
        self.BLOCK_WIDTH = (self.WIDTH - 2 * self.GRID_SIDE_MARGIN) // self.BLOCK_COLS
        self.BLOCK_HEIGHT = 18
        self.BLOCK_SPACING = 2
        self.MAX_EPISODE_STEPS = 1500
        self.MAX_AIM_ANGLE_CHANGE = 0.05

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.balls_left = 0
        self.total_blocks_destroyed = 0
        self.game_phase = 'AIMING'
        self.aim_angle = 0
        self.launcher_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 30)
        self.active_balls = []
        self.particles = []
        self.blocks = []
        self.prev_space_state = 0
        self.blocks_destroyed_this_launch = 0
        self.blocks_hit_this_launch = set()

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.balls_left = 3
        self.total_blocks_destroyed = 0
        
        self.game_phase = 'AIMING'
        self.aim_angle = -math.pi / 2
        
        self.active_balls.clear()
        self.particles.clear()
        self._create_blocks()

        self.prev_space_state = 0
        self.blocks_destroyed_this_launch = 0
        self.blocks_hit_this_launch.clear()
        
        return self._get_observation(), self._get_info()
    
    def _create_blocks(self):
        self.blocks.clear()
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - grid_width) / 2
        
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                health = 1 + r // 2 # Top rows are tougher
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.GRID_TOP_MARGIN + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                
                block = {
                    "id": r * self.BLOCK_COLS + c,
                    "rect": rect,
                    "health": health,
                    "max_health": health,
                }
                self.blocks.append(block)

    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            self._handle_input(action)
            reward += self._update_physics()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.game_won:
            reward += 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.game_phase == 'AIMING':
            # Aiming
            if movement == 3: # Left
                self.aim_angle -= self.MAX_AIM_ANGLE_CHANGE
            elif movement == 4: # Right
                self.aim_angle += self.MAX_AIM_ANGLE_CHANGE
            
            # Clamp angle to avoid horizontal shots
            self.aim_angle = max(-math.pi + 0.1, min(-0.1, self.aim_angle))

            # Launching (on key press)
            if space_held and not self.prev_space_state and self.balls_left > 0:
                self.game_phase = 'BALL_IN_PLAY'
                self.balls_left -= 1
                self.blocks_destroyed_this_launch = 0
                self.blocks_hit_this_launch.clear()
                
                vel = pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * self.BALL_SPEED
                pos = self.launcher_pos + vel * 2 # Start slightly off the launcher
                
                ball = {"pos": pos, "vel": vel, "radius": self.BALL_RADIUS}
                self.active_balls.append(ball)
                # sfx: ball_launch.wav

        self.prev_space_state = space_held

    def _update_physics(self):
        step_reward = 0.0
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # Update balls
        for ball in self.active_balls[:]:
            ball['pos'] += ball['vel']

            # Wall collisions
            if ball['pos'].x - ball['radius'] < 0 or ball['pos'].x + ball['radius'] > self.WIDTH:
                ball['vel'].x *= -1
                ball['pos'].x = max(ball['radius'], min(self.WIDTH - ball['radius'], ball['pos'].x))
                # sfx: bounce_wall.wav
            if ball['pos'].y - ball['radius'] < 0:
                ball['vel'].y *= -1
                ball['pos'].y = ball['radius']
                # sfx: bounce_wall.wav

            # Ball out of bounds (bottom)
            if ball['pos'].y > self.HEIGHT:
                self.active_balls.remove(ball)
                continue

            # Block collisions
            ball_rect = pygame.Rect(ball['pos'].x - ball['radius'], ball['pos'].y - ball['radius'], ball['radius']*2, ball['radius']*2)
            for block in self.blocks[:]:
                if ball_rect.colliderect(block['rect']):
                    # sfx: bounce_block.wav
                    
                    # Determine collision side to correctly reflect velocity
                    prev_pos = ball['pos'] - ball['vel']
                    if (prev_pos.y - ball['radius'] > block['rect'].bottom or
                        prev_pos.y + ball['radius'] < block['rect'].top):
                        ball['vel'].y *= -1
                    else:
                        ball['vel'].x *= -1

                    block['health'] -= 1
                    
                    if block['id'] not in self.blocks_hit_this_launch:
                        step_reward += 0.1
                        self.blocks_hit_this_launch.add(block['id'])

                    if block['health'] <= 0:
                        # sfx: block_destroy.wav
                        self._spawn_particles(block['rect'].center, self.PARTICLE_COLORS[block['max_health'] % len(self.PARTICLE_COLORS)])
                        self.blocks.remove(block)
                        step_reward += 1.0
                        self.blocks_destroyed_this_launch += 1
                        self.total_blocks_destroyed += 1
                        self.score = self.total_blocks_destroyed * 10
                    break # Only collide with one block per frame
        
        # Check for phase transition
        if self.game_phase == 'BALL_IN_PLAY' and not self.active_balls:
            if self.blocks_destroyed_this_launch > 1:
                step_reward += 5.0 # Combo bonus
            
            if self.balls_left > 0 and self.total_blocks_destroyed < self.BLOCK_ROWS * self.BLOCK_COLS:
                self.game_phase = 'AIMING'
            else:
                self.game_over = True

        return step_reward

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            particle = {
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": random.randint(15, 30),
                "color": color,
                "size": random.randint(2, 4)
            }
            self.particles.append(particle)
            
    def _check_termination(self):
        if not self.blocks:
            self.game_won = True
            self.game_over = True
            return True
        if self.game_over: # True if balls run out and last ball is gone
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            color_index = min(len(self.BLOCK_COLORS) - 1, block['max_health'] - 1)
            base_color = self.BLOCK_COLORS[color_index]
            
            health_ratio = block['health'] / block['max_health']
            # Interpolate color to a darker shade for damage
            final_color = (
                int(base_color[0] * (0.5 + 0.5 * health_ratio)),
                int(base_color[1] * (0.5 + 0.5 * health_ratio)),
                int(base_color[2] * (0.5 + 0.5 * health_ratio))
            )
            pygame.draw.rect(self.screen, final_color, block['rect'])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Draw launcher platform
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER, (self.launcher_pos.x - 20, self.launcher_pos.y - 5, 40, 10), border_radius=3)
        
        # Draw aiming line
        if self.game_phase == 'AIMING':
            end_pos = self.launcher_pos + pygame.Vector2(math.cos(self.aim_angle), math.sin(self.aim_angle)) * 70
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, self.launcher_pos, end_pos, 2)

        # Draw balls
        for ball in self.active_balls:
            x, y, r = int(ball['pos'].x), int(ball['pos'].y), ball['radius']
            # Glow effect
            glow_surf = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 255, 255, 50), (r*2, r*2), r*2)
            self.screen.blit(glow_surf, (x - r*2, y - r*2))
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_BALL)

    def _render_ui(self):
        # Draw score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))

        # Draw balls left
        balls_text = self.font_ui.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (15, 10))
        for i in range(self.balls_left):
            x = 90 + i * 20
            y = 10 + self.font_ui.get_height() // 2
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw game over/win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            msg_text = self.font_msg.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "game_phase": self.game_phase,
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- To run and play the game manually ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a separate display for manual play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    while running:
        # --- Action mapping for human player ---
        keys = pygame.key.get_pressed()
        move = 0 # no-op
        if keys[pygame.K_LEFT]: move = 3
        elif keys[pygame.K_RIGHT]: move = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Display the observation ---
        # The observation is (H, W, C), but pygame blit needs (W, H) surface
        # The env._get_observation already creates the surface, we can just use it
        display_screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()