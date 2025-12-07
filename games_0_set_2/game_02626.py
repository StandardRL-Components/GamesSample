
# Generated: 2025-08-27T20:57:17.345823
# Source Brief: brief_02626.md
# Brief Index: 2626

        
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


class IsometricConverter:
    """A helper class for isometric coordinate conversions and drawing."""
    def __init__(self, screen_width, screen_height, grid_width, tile_size):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_width = grid_width
        self.tile_size = tile_size
        self.tile_width_half = self.tile_size
        self.tile_height_half = self.tile_size / 2
        self.origin_x = screen_width / 2
        self.origin_y = screen_height / 4

    def world_to_screen(self, x, y, z=0):
        """Converts 3D world grid coordinates to 2D screen pixel coordinates."""
        screen_x = self.origin_x + (x - y) * self.tile_width_half
        screen_y = self.origin_y + (x + y) * self.tile_height_half - z * self.tile_size
        return int(screen_x), int(screen_y)

    def draw_iso_cube(self, surface, pos, colors, height=1):
        """Draws an isometric cube on the given surface."""
        x, y = pos
        top_color, side_color_1, side_color_2 = colors
        
        # Calculate vertices for the top face
        p1 = self.world_to_screen(x, y, height)
        p2 = self.world_to_screen(x + 1, y, height)
        p3 = self.world_to_screen(x + 1, y + 1, height)
        p4 = self.world_to_screen(x, y + 1, height)
        
        # Draw side faces first (for correct layering)
        pygame.gfxdraw.filled_polygon(surface, [p2, self.world_to_screen(x + 1, y, 0), self.world_to_screen(x + 1, y + 1, 0), p3], side_color_1)
        pygame.gfxdraw.filled_polygon(surface, [p4, self.world_to_screen(x, y + 1, 0), self.world_to_screen(x + 1, y + 1, 0), p3], side_color_2)
        
        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, [p1, p2, p3, p4], top_color)
        
        # Draw anti-aliased outlines for a clean look
        pygame.gfxdraw.aapolygon(surface, [p1, p2, p3, p4], top_color)
        pygame.gfxdraw.aapolygon(surface, [p2, self.world_to_screen(x + 1, y, 0), self.world_to_screen(x + 1, y + 1, 0), p3], side_color_1)
        pygame.gfxdraw.aapolygon(surface, [p4, self.world_to_screen(x, y + 1, 0), self.world_to_screen(x + 1, y + 1, 0), p3], side_color_2)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Press space to place a block."
    )

    game_description = (
        "Guide the falling ball to the green target by placing blocks. Don't let it fall off or run out of time!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 16
        self.TILE_SIZE = 18

        # Game parameters
        self.MAX_STEPS = 1800  # 60 seconds at 30 FPS
        self.MAX_BLOCKS = 15
        self.GRAVITY = 0.015
        self.BOUNCE_FACTOR = 0.7
        self.BALL_RADIUS_GRID = 0.4 # Ball radius in grid units

        # Colors
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_WALL = ((80, 85, 90), (70, 75, 80), (60, 65, 70))
        self.COLOR_BLOCK = ((180, 180, 180), (160, 160, 160), (140, 140, 140))
        self.COLOR_CURSOR = (100, 200, 255, 100)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_TARGET = (50, 200, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TIMER_NORMAL = (255, 200, 0)
        self.COLOR_TIMER_WARN = (255, 50, 50)
        
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
        self.font_msg = pygame.font.Font(None, 60)
        
        # Game helpers
        self.iso_converter = IsometricConverter(self.WIDTH, self.HEIGHT, self.GRID_WIDTH, self.TILE_SIZE)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = 0
        self.blocks_remaining = 0
        self.placed_blocks = []
        self.ball_pos = pygame.Vector3(0, 0, 0)
        self.ball_vel = pygame.Vector3(0, 0, 0)
        self.cursor_pos = pygame.Vector2(0, 0)
        self.last_space_held = False
        self.particles = []
        self.maze_walls = []
        self.target_pos = (0,0)

        # Initialize state variables
        self.reset()

    def _generate_maze(self):
        self.maze_walls = []
        for i in range(self.GRID_WIDTH):
            self.maze_walls.append((i, -1))
            self.maze_walls.append((-1, i))
        
        # Some example pillars
        self.maze_walls.extend([(4, 4), (4, 5), (5, 4), (10, 10), (10, 11), (11, 10)])
        self.maze_walls.extend([(2, 8), (3, 8), (4, 8), (5, 8)])
        self.maze_walls.extend([(12, 3), (12, 4), (12, 5)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()
        self.target_pos = (self.GRID_WIDTH - 3, self.GRID_HEIGHT - 3)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_left = self.MAX_STEPS
        self.blocks_remaining = self.MAX_BLOCKS
        
        self.placed_blocks = []
        self.particles = []

        self.ball_pos = pygame.Vector3(2.5, 2.5, 10)
        self.ball_vel = pygame.Vector3(0.01, 0.01, 0) # Small initial push
        
        self.cursor_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def _create_particles(self, pos, color, count=10):
        # sound: bounce_effect.wav
        for _ in range(count):
            vel = pygame.Vector3(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(0.1, 0.5))
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': random.randint(10, 20), 'color': color})

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        
        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # 1. Handle player input
            # Cursor movement
            if movement == 1: self.cursor_pos.y -= 1  # Up
            elif movement == 2: self.cursor_pos.y += 1  # Down
            elif movement == 3: self.cursor_pos.x -= 1  # Left
            elif movement == 4: self.cursor_pos.x += 1  # Right
            self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.GRID_WIDTH - 1)
            self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.GRID_HEIGHT - 1)

            # Block placement
            if space_held and not self.last_space_held and self.blocks_remaining > 0:
                is_valid_placement = True
                new_block_pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
                if new_block_pos in [b['pos'] for b in self.placed_blocks]: is_valid_placement = False
                if new_block_pos in self.maze_walls: is_valid_placement = False
                if new_block_pos == self.target_pos: is_valid_placement = False
                
                if is_valid_placement:
                    # sound: place_block.wav
                    self.placed_blocks.append({'pos': new_block_pos, 'reward_pending': -1.0})
                    self.blocks_remaining -= 1
                    reward -= 1.0 # Penalize placing a block, will be refunded if useful
                    pos3d = pygame.Vector3(new_block_pos[0] + 0.5, new_block_pos[1] + 0.5, 1)
                    self._create_particles(pos3d, self.COLOR_BLOCK[0], count=15)

            self.last_space_held = space_held

            # 2. Update ball physics
            prev_ball_pos = self.ball_pos.copy()
            self.ball_vel.z -= self.GRAVITY
            self.ball_pos += self.ball_vel

            # 3. Collision detection
            solids = self.maze_walls + [b['pos'] for b in self.placed_blocks]

            # Floor collision
            if self.ball_pos.z - self.BALL_RADIUS_GRID < 0 and self.ball_vel.z < 0:
                self.ball_pos.z = self.BALL_RADIUS_GRID
                self.ball_vel.z *= -self.BOUNCE_FACTOR
                if abs(self.ball_vel.z) < 0.05: self.ball_vel.z = 0 # Rest
                self._create_particles(self.ball_pos, self.COLOR_GRID, 3)

            # Solid blocks/walls collision
            for solid_pos in solids:
                if solid_pos[0] < self.ball_pos.x < solid_pos[0] + 1 and \
                   solid_pos[1] < self.ball_pos.y < solid_pos[1] + 1:
                    if prev_ball_pos.z >= 1 and self.ball_pos.z < 1 and self.ball_vel.z < 0:
                        self.ball_pos.z = 1
                        self.ball_vel.z *= -self.BOUNCE_FACTOR
                        self._create_particles(self.ball_pos, self.COLOR_BALL, 5)
                        
                        for block in self.placed_blocks:
                            if block['pos'] == solid_pos and block['reward_pending'] != 0:
                                reward += 6.0 # +5 useful block, +1 refund placement cost
                                block['reward_pending'] = 0
                                break
            
            for solid_pos in solids:
                if 0 < self.ball_pos.z < 1:
                    if solid_pos[1] < self.ball_pos.y < solid_pos[1] + 1:
                        if prev_ball_pos.x >= solid_pos[0] + 1 and self.ball_pos.x < solid_pos[0] + 1:
                            self.ball_pos.x = solid_pos[0] + 1
                            self.ball_vel.x *= -self.BOUNCE_FACTOR
                        elif prev_ball_pos.x <= solid_pos[0] and self.ball_pos.x > solid_pos[0]:
                            self.ball_pos.x = solid_pos[0]
                            self.ball_vel.x *= -self.BOUNCE_FACTOR
                    if solid_pos[0] < self.ball_pos.x < solid_pos[0] + 1:
                        if prev_ball_pos.y >= solid_pos[1] + 1 and self.ball_pos.y < solid_pos[1] + 1:
                            self.ball_pos.y = solid_pos[1] + 1
                            self.ball_vel.y *= -self.BOUNCE_FACTOR
                        elif prev_ball_pos.y <= solid_pos[1] and self.ball_pos.y > solid_pos[1]:
                            self.ball_pos.y = solid_pos[1]
                            self.ball_vel.y *= -self.BOUNCE_FACTOR

            # 4. Update game state
            self.steps += 1
            self.time_left -= 1
            reward += 0.1 # Survival reward

            self.particles = [p for p in self.particles if p['life'] > 0]
            for p in self.particles:
                p['pos'] += p['vel']
                p['vel'].z -= self.GRAVITY * 0.1
                p['life'] -= 1

        # 5. Check termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100.0
            else:
                dist_to_target = math.hypot(self.ball_pos.x - (self.target_pos[0] + 0.5), self.ball_pos.y - (self.target_pos[1] + 0.5))
                if self.time_left <= 0:
                    reward -= 50.0 # Timeout
                else:
                    reward -= 100.0 # Fell off

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.game_over:
            return True

        dist_to_target = math.hypot(self.ball_pos.x - (self.target_pos[0] + 0.5), self.ball_pos.y - (self.target_pos[1] + 0.5))
        if dist_to_target < 0.5 and self.ball_pos.z < 0.1:
            # sound: win.wav
            self.game_over = True
            self.win = True
            return True
        elif self.ball_pos.z < -2 or not (0 <= self.ball_pos.x < self.GRID_WIDTH and 0 <= self.ball_pos.y < self.GRID_HEIGHT):
            # sound: lose.wav
            self.game_over = True
            return True
        elif self.time_left <= 0:
            # sound: timeout.wav
            self.game_over = True
            return True
        return False
        
    def _calculate_reward(self):
        # This logic is integrated into the step function for clarity
        return 0

    def _render_game(self):
        # Draw grid floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self.iso_converter.world_to_screen(x, y)
                p2 = self.iso_converter.world_to_screen(x + 1, y)
                p3 = self.iso_converter.world_to_screen(x + 1, y + 1)
                p4 = self.iso_converter.world_to_screen(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Draw target
        tx, ty = self.target_pos
        center_screen = self.iso_converter.world_to_screen(tx + 0.5, ty + 0.5)
        pygame.gfxdraw.filled_circle(self.screen, center_screen[0], center_screen[1], int(self.TILE_SIZE * 0.8), self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, center_screen[0], center_screen[1], int(self.TILE_SIZE * 0.8), self.COLOR_TARGET)
        
        # Draw ball shadow
        shadow_pos = self.iso_converter.world_to_screen(self.ball_pos.x, self.ball_pos.y, 0)
        shadow_radius = int(self.TILE_SIZE * self.BALL_RADIUS_GRID * (1 - min(1, self.ball_pos.z / 15.0)))
        if shadow_radius > 0:
            shadow_surf = pygame.Surface((shadow_radius*2, shadow_radius*2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, shadow_surf.get_rect())
            self.screen.blit(shadow_surf, (shadow_pos[0] - shadow_radius, shadow_pos[1] - shadow_radius))

        # Draw elements in correct Z-order (back to front)
        draw_queue = []
        for wall_pos in self.maze_walls:
            draw_queue.append({'type': 'wall', 'pos': wall_pos, 'y': wall_pos[1]})
        for block in self.placed_blocks:
            draw_queue.append({'type': 'block', 'pos': block['pos'], 'y': block['pos'][1]})
        
        draw_queue.append({'type': 'ball', 'y': self.ball_pos.y})
        draw_queue.append({'type': 'cursor', 'y': self.cursor_pos.y})
        
        for p in self.particles:
            draw_queue.append({'type': 'particle', 'particle': p, 'y': p['pos'].y})

        draw_queue.sort(key=lambda item: item['y'])

        for item in draw_queue:
            if item['type'] == 'wall':
                self.iso_converter.draw_iso_cube(self.screen, item['pos'], self.COLOR_WALL)
            elif item['type'] == 'block':
                self.iso_converter.draw_iso_cube(self.screen, item['pos'], self.COLOR_BLOCK)
            elif item['type'] == 'cursor':
                c_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                self.iso_converter.draw_iso_cube(c_surf, self.cursor_pos, (self.COLOR_CURSOR, self.COLOR_CURSOR, self.COLOR_CURSOR))
                self.screen.blit(c_surf, (0,0))
            elif item['type'] == 'ball':
                ball_screen_pos = self.iso_converter.world_to_screen(self.ball_pos.x, self.ball_pos.y, self.ball_pos.z)
                ball_radius_px = int(self.TILE_SIZE * self.BALL_RADIUS_GRID)
                pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], ball_radius_px, self.COLOR_BALL)
                pygame.gfxdraw.aacircle(self.screen, ball_screen_pos[0], ball_screen_pos[1], ball_radius_px, self.COLOR_BALL)
            elif item['type'] == 'particle':
                p = item['particle']
                p_pos = self.iso_converter.world_to_screen(p['pos'].x, p['pos'].y, p['pos'].z)
                alpha = max(0, 255 * (p['life'] / 20.0))
                color = (*p['color'][:3], int(alpha))
                pygame.draw.circle(self.screen, color, p_pos, 2)


    def _render_ui(self):
        time_percent = self.time_left / self.MAX_STEPS
        timer_color = self.COLOR_TIMER_NORMAL if time_percent > 0.2 else self.COLOR_TIMER_WARN
        time_text = f"TIME: {max(0, self.time_left // 30):02d}"
        time_surf = self.font_ui.render(time_text, True, timer_color)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        block_text = f"BLOCKS: {self.blocks_remaining}/{self.MAX_BLOCKS}"
        block_surf = self.font_ui.render(block_text, True, self.COLOR_TEXT)
        self.screen.blit(block_surf, (10, 10))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_TARGET if self.win else self.COLOR_TIMER_WARN
            msg_surf = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Block Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    running = True
    while running:
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(2000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()