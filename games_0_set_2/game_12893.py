import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:27:10.172047
# Source Brief: brief_02893.md
# Brief Index: 2893
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque, defaultdict

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Match 3 or more hexagonal tiles of the same color to score points in this fast-paced puzzle game."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place a tile. Press shift to cycle to the next tile color."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.HEX_RADIUS = 18
        self.GRID_ROWS = 9
        self.GRID_COLS = 15
        self.MAX_SCORE = 1000
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS # Match steps to time limit for auto-advance

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 160, 80),  # Orange
        ]
        self.POWERUP_COLOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_SHADOW = (10, 10, 15)

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
        self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
        self.font_medium = pygame.font.SysFont('Consolas', 24, bold=True)
        
        # Hex grid setup
        self._hex_centers = {}
        self._grid_coords = set()
        self._initialize_grid_layout()

        # Initialize state variables
        self.grid = None
        self.cursor_q = None
        self.cursor_r = None
        self.cursor_vis_pos = None
        self.current_tile_type = None
        self.score = None
        self.steps = None
        self.time_remaining_frames = None
        self.game_over = None
        self.last_space_held = None
        self.last_shift_held = None
        self.particles = None
        self.animations = None
        
        # self.reset() is called by the environment wrapper
        
    def _initialize_grid_layout(self):
        w = 2 * self.HEX_RADIUS
        h = math.sqrt(3) * self.HEX_RADIUS
        offset_x = self.SCREEN_WIDTH / 2
        offset_y = self.SCREEN_HEIGHT / 2 + 20
        
        for r in range(-(self.GRID_ROWS // 2), self.GRID_ROWS // 2 + 1):
            r_offset = r // 2
            for q in range(-(self.GRID_COLS // 2) - r_offset, self.GRID_COLS // 2 - r_offset + 1):
                x = offset_x + w * (q + r / 2)
                y = offset_y + h * r * 3/4
                if 0 < x < self.SCREEN_WIDTH and 0 < y < self.SCREEN_HEIGHT - 60:
                    self._hex_centers[(q, r)] = (x, y)
                    self._grid_coords.add((q, r))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = defaultdict(lambda: -1) # -1 is empty
        self.cursor_q, self.cursor_r = 0, 0
        
        target_pos = self._axial_to_pixel(self.cursor_q, self.cursor_r)
        self.cursor_vis_pos = np.array(target_pos, dtype=float)

        self.current_tile_type = self.np_random.integers(0, len(self.TILE_COLORS))
        self.score = 0
        self.steps = 0
        self.time_remaining_frames = self.MAX_TIME_SECONDS * self.FPS
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False

        self.particles = []
        self.animations = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        self.time_remaining_frames -= 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # --- Handle Actions ---
        if shift_pressed:
            self.current_tile_type = (self.current_tile_type + 1) % len(self.TILE_COLORS)
            # SFX: cycle_tile_type.wav
        
        self._move_cursor(movement)

        if space_pressed:
            if self.grid[(self.cursor_q, self.cursor_r)] == -1:
                # SFX: place_tile.wav
                self.grid[(self.cursor_q, self.cursor_r)] = self.current_tile_type
                reward += 1 # Placement reward
                
                chain_reaction_reward = self._resolve_matches()
                reward += chain_reaction_reward
                
                self.current_tile_type = self.np_random.integers(0, len(self.TILE_COLORS))

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- Update Game State ---
        self._update_animations()
        self._update_particles()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated and self.score >= self.MAX_SCORE:
            reward += 100 # Victory reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _resolve_matches(self):
        total_reward = 0
        
        while True:
            all_matches = self._find_all_matches()
            if not all_matches:
                break
            
            powerup_locations = []
            all_matched_coords = set()

            for match in all_matches:
                all_matched_coords.update(match)
                
                # Calculate match reward
                size = len(match)
                if size == 3: total_reward += 10
                elif size == 4: total_reward += 20
                elif size == 5: total_reward += 30
                else: total_reward += 40
                self.score += size * 10
                
                # Check for power-up creation
                if size >= 4:
                    powerup_locations.append(random.choice(list(match)))

            # Remove matched tiles
            for q, r in all_matched_coords:
                self._create_particles(q, r, self.grid[(q, r)])
                self._create_animation(q, r, 'match_flash')
                self.grid[(q, r)] = -1
                # SFX: match_clear.wav

            # Activate power-ups
            if powerup_locations:
                total_reward += len(powerup_locations) * 5
                for q, r in powerup_locations:
                    # SFX: powerup_activate.wav
                    self._create_animation(q, r, 'powerup_blast')
                    neighbors = self._get_neighbors(q, r)
                    for nq, nr in neighbors:
                        if (nq, nr) in self.grid and self.grid[(nq, nr)] != -1:
                            self.grid[(nq, nr)] = (self.grid[(nq, nr)] + 1) % len(self.TILE_COLORS)
                            self._create_animation(nq, nr, 'tile_change')
        
        return total_reward

    def _find_all_matches(self):
        matches = []
        visited = set()
        for q, r in self._grid_coords:
            if self.grid[(q, r)] != -1 and (q, r) not in visited:
                color = self.grid[(q, r)]
                component = self._flood_fill(q, r, color)
                visited.update(component)
                if len(component) >= 3:
                    matches.append(component)
        return matches

    def _flood_fill(self, start_q, start_r, target_color):
        if self.grid[(start_q, start_r)] != target_color:
            return set()
            
        q = deque([(start_q, start_r)])
        component = set([(start_q, start_r)])
        
        while q:
            curr_q, curr_r = q.popleft()
            for nq, nr in self._get_neighbors(curr_q, curr_r):
                if (nq, nr) in self._grid_coords and (nq, nr) not in component and self.grid[(nq, nr)] == target_color:
                    component.add((nq, nr))
                    q.append((nq, nr))
        return component

    def _move_cursor(self, movement):
        q, r = self.cursor_q, self.cursor_r
        if movement == 1: # Up
            q, r = q, r - 1
        elif movement == 2: # Down
            q, r = q, r + 1
        elif movement == 3: # Left
            q, r = q - 1, r
        elif movement == 4: # Right
            q, r = q + 1, r
        
        if (q, r) in self._grid_coords:
            self.cursor_q, self.cursor_r = q, r

    def _check_termination(self):
        if self.score >= self.MAX_SCORE or self.time_remaining_frames <= 0:
            self.game_over = True
            return True
        return False

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
            "time_remaining_seconds": self.time_remaining_frames / self.FPS,
            "cursor_pos": (self.cursor_q, self.cursor_r),
        }

    def _render_game(self):
        # Draw base grid
        for q, r in self._grid_coords:
            center = self._hex_centers[(q, r)]
            self._draw_hexagon(self.screen, self.COLOR_GRID, center, self.HEX_RADIUS, width=1)
        
        # Draw placed tiles
        for (q, r), tile_type in self.grid.items():
            if tile_type != -1:
                center = self._hex_centers[(q, r)]
                color = self.TILE_COLORS[tile_type]
                self._draw_hexagon(self.screen, color, center, self.HEX_RADIUS)
        
        # Draw animations
        for anim in self.animations:
            anim['func'](anim)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw cursor
        target_pos = self._axial_to_pixel(self.cursor_q, self.cursor_r)
        self.cursor_vis_pos = self.cursor_vis_pos * 0.6 + np.array(target_pos) * 0.4
        self._draw_hexagon(self.screen, self.COLOR_CURSOR, self.cursor_vis_pos, self.HEX_RADIUS + 2, width=3)
        
    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_large, (20, 15))
        
        # Timer
        time_sec = max(0, int(self.time_remaining_frames / self.FPS))
        time_text = f"TIME: {time_sec:02d}"
        text_width = self.font_large.size(time_text)[0]
        self._draw_text(time_text, self.font_large, (self.SCREEN_WIDTH - text_width - 20, 15))
        
        # Next tile display
        self._draw_text("NEXT", self.font_medium, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50), center=True)
        next_tile_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25)
        self._draw_hexagon(self.screen, self.TILE_COLORS[self.current_tile_type], next_tile_pos, self.HEX_RADIUS)
        self._draw_hexagon(self.screen, self.COLOR_UI_TEXT, next_tile_pos, self.HEX_RADIUS, width=2)

    def _draw_text(self, text, font, pos, color=None, shadow=True, center=False):
        if color is None: color = self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        if shadow:
            shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
            shadow_rect = shadow_surface.get_rect()
            shadow_rect.topleft = (text_rect.left + 2, text_rect.top + 2)
            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surface, shadow_rect)
        
        self.screen.blit(text_surface, text_rect)

    def _draw_hexagon(self, surface, color, center, radius, width=0):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((int(x), int(y)))
        
        if width == 0:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _axial_to_pixel(self, q, r):
        return self._hex_centers.get((q, r), (0, 0))

    def _get_neighbors(self, q, r):
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if (nq, nr) in self._grid_coords:
                neighbors.append((nq, nr))
        return neighbors

    def _create_particles(self, q, r, tile_type):
        center = self._axial_to_pixel(q, r)
        color = self.TILE_COLORS[tile_type]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': np.array(center, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            p['radius'] *= 0.97
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0.5]

    def _create_animation(self, q, r, anim_type):
        pos = self._axial_to_pixel(q, r)
        if anim_type == 'match_flash':
            self.animations.append({'type': anim_type, 'pos': pos, 'life': 10, 'func': self._render_match_flash})
        elif anim_type == 'powerup_blast':
            self.animations.append({'type': anim_type, 'pos': pos, 'life': 20, 'func': self._render_powerup_blast})
        elif anim_type == 'tile_change':
            self.animations.append({'type': anim_type, 'pos': pos, 'life': 15, 'func': self._render_tile_change})
    
    def _update_animations(self):
        for anim in self.animations:
            anim['life'] -= 1
        self.animations = [a for a in self.animations if a['life'] > 0]

    def _render_match_flash(self, anim):
        progress = anim['life'] / 10.0
        alpha = int(255 * math.sin(progress * math.pi))
        radius = self.HEX_RADIUS * (1.2 - 0.2 * progress)
        color = (255, 255, 255, alpha)
        
        temp_surf = pygame.Surface((int(radius*2), int(radius*2)), pygame.SRCALPHA)
        self._draw_hexagon(temp_surf, color, (radius, radius), radius)
        self.screen.blit(temp_surf, (anim['pos'][0] - radius, anim['pos'][1] - radius))

    def _render_powerup_blast(self, anim):
        progress = 1.0 - (anim['life'] / 20.0)
        radius = self.HEX_RADIUS * 2.5 * progress
        alpha = int(255 * (1.0 - progress))
        
        if radius > 1:
            pygame.gfxdraw.aacircle(self.screen, int(anim['pos'][0]), int(anim['pos'][1]), int(radius), (255, 255, 255, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(anim['pos'][0]), int(anim['pos'][1]), int(radius-1), (255, 255, 255, alpha))

    def _render_tile_change(self, anim):
        progress = anim['life'] / 15.0
        alpha = int(255 * math.sin(progress * math.pi))
        radius = self.HEX_RADIUS * (1.0 + 0.5 * (1.0 - progress))
        color = (255, 255, 255, alpha)

        temp_surf = pygame.Surface((int(radius*2), int(radius*2)), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (radius, radius), radius)
        self.screen.blit(temp_surf, (anim['pos'][0] - radius, anim['pos'][1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()
        
    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # Example of how to use the environment
    # This part is for human play and visualization, and will not be used by the evaluation server.
    # It is recommended to run this code to test the environment before submission.
    
    # Un-comment the following line to run in a window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("HexaMatch Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map keyboard keys to actions for human play
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            obs, info = env.reset()
            total_reward = 0
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata['render_fps'])

    env.close()