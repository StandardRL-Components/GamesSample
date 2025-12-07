import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. ↑↓ to rotate. Space for soft drop, Shift for hard drop."
    )

    game_description = (
        "An isometric puzzle game. Place falling jewel pairs to create matches of 3 or more. "
        "Clear 20 groups of jewels to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_D, self.GRID_H = 8, 8, 16
        self.MAX_STEPS = 2500
        self.WIN_CONDITION = 20 # Number of match groups to clear

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_SHADOW = (10, 10, 15)
        self.JEWEL_COLORS = [
            (255, 50, 50),   # 1: Red
            (50, 255, 50),   # 2: Green
            (80, 80, 255),   # 3: Blue
            (255, 255, 80),  # 4: Yellow
            (255, 120, 50),  # 5: Orange
            (200, 50, 200),  # 6: Purple
            (50, 200, 200),  # 7: Cyan
        ]

        # Isometric projection constants
        self.TILE_W_HALF = 16
        self.TILE_H_HALF = 8
        self.TILE_Z_HEIGHT = 12
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # Gameplay constants
        self.MOVE_COOLDOWN = 4  # frames
        self.ROTATE_COOLDOWN = 6 # frames
        self.GRAVITY_RATE = 20 # frames per step down
        self.SOFT_DROP_MULTIPLIER = 5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.steps = None
        self.score = None
        self.match_groups_cleared = None
        self.game_over = None
        self.np_random = None

        self.piece = None
        self.next_piece = None
        
        self.move_timer = None
        self.rotate_timer = None
        self.gravity_timer = None

        self.particles = None
        self.flash_effects = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_H, self.GRID_D, self.GRID_W), dtype=np.int8)
        self.steps = 0
        self.score = 0
        self.match_groups_cleared = 0
        self.game_over = False

        self.move_timer = 0
        self.rotate_timer = 0
        self.gravity_timer = 0
        
        self.particles = []
        self.flash_effects = []

        self.piece = None
        self.next_piece = None
        
        self._spawn_piece() # Spawns next_piece
        self._spawn_piece() # Spawns current piece, moves next_piece up

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self._update_timers()
        self._update_effects()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            reward += self._handle_hard_drop()
        else:
            self._handle_input(movement)
            reward += self._apply_gravity_and_placing(space_held)
        
        if self.game_over:
            reward -= 100 # Loss penalty
            terminated = True
        elif self.match_groups_cleared >= self.WIN_CONDITION:
            reward += 100 # Win bonus
            self.game_over = True
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    # --- Private Game Logic Methods ---
    
    def _update_timers(self):
        self.move_timer = max(0, self.move_timer - 1)
        self.rotate_timer = max(0, self.rotate_timer - 1)
        self.gravity_timer += 1

    def _handle_input(self, movement):
        # 0=none, 1=up(rotCW), 2=down(rotCCW), 3=left, 4=right
        if self.piece is None: return

        if movement in [3, 4] and self.move_timer == 0:
            dx = -1 if movement == 3 else 1
            if not self._check_collision(dx, 0, 0, 0):
                self.piece['pos'][0] += dx
                self.move_timer = self.MOVE_COOLDOWN
        
        if movement in [1, 2] and self.rotate_timer == 0:
            d_rot = 1 if movement == 1 else -1
            if not self._check_collision(0, 0, 0, d_rot):
                self.piece['rot'] = (self.piece['rot'] + d_rot) % 4
                self.rotate_timer = self.ROTATE_COOLDOWN

    def _apply_gravity_and_placing(self, soft_drop):
        if self.piece is None: return 0

        fall_speed = self.GRAVITY_RATE
        if soft_drop:
            fall_speed //= self.SOFT_DROP_MULTIPLIER

        reward = 0
        if self.gravity_timer >= fall_speed:
            self.gravity_timer = 0
            if self._check_collision(0, 0, -1, 0):
                reward += self._place_piece()
            else:
                self.piece['pos'][2] -= 1
        return reward

    def _handle_hard_drop(self):
        if self.piece is None: return 0
        # Find landing spot
        while not self._check_collision(0, 0, -1, 0):
            self.piece['pos'][2] -= 1
        # Place piece and get rewards
        return self._place_piece()

    def _place_piece(self):
        # Sound: Block land
        reward = 0.1
        self.score += 1
        
        coords = self._get_piece_coords()
        jewel1_coords, jewel2_coords = coords[0], coords[1]
        
        if 0 <= jewel1_coords[0] < self.GRID_W and 0 <= jewel1_coords[1] < self.GRID_D and 0 <= jewel1_coords[2] < self.GRID_H:
             self.grid[jewel1_coords[2], jewel1_coords[1], jewel1_coords[0]] = self.piece['colors'][0]
        if 0 <= jewel2_coords[0] < self.GRID_W and 0 <= jewel2_coords[1] < self.GRID_D and 0 <= jewel2_coords[2] < self.GRID_H:
             self.grid[jewel2_coords[2], jewel2_coords[1], jewel2_coords[0]] = self.piece['colors'][1]

        chain_level = 1
        while True:
            match_reward, cleared_count, cleared_groups = self._find_and_clear_matches()
            if cleared_count == 0:
                break
            
            reward += match_reward * chain_level
            self.score += cleared_count * 10 * chain_level
            self.match_groups_cleared += cleared_groups
            self._settle_grid()
            chain_level += 1
        
        self._spawn_piece()
        if self.piece and self._check_collision(0, 0, 0, 0):
            self.game_over = True

        return reward

    def _find_and_clear_matches(self):
        to_clear = set()
        visited_for_match = set()
        
        for z in range(self.GRID_H):
            for y in range(self.GRID_D):
                for x in range(self.GRID_W):
                    if self.grid[z, y, x] != 0 and (x, y, z) not in visited_for_match:
                        color = self.grid[z, y, x]
                        group = set()
                        q = deque([(x, y, z)])
                        visited_this_group = set([(x,y,z)])
                        
                        while q:
                            cx, cy, cz = q.popleft()
                            group.add((cx, cy, cz))
                            visited_for_match.add((cx, cy, cz))
                            
                            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                                nx, ny, nz = cx + dx, cy + dy, cz + dz
                                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_D and 0 <= nz < self.GRID_H and \
                                   self.grid[nz, ny, nx] == color and (nx, ny, nz) not in visited_this_group:
                                    visited_this_group.add((nx,ny,nz))
                                    q.append((nx,ny,nz))
                        
                        if len(group) >= 3:
                            to_clear.update(group)

        if not to_clear:
            return 0, 0, 0

        # Sound: Match clear
        reward = 0
        if len(to_clear) == 3: reward = 1
        elif len(to_clear) == 4: reward = 5
        else: reward = 10
        
        for x, y, z in to_clear:
            color = self.JEWEL_COLORS[self.grid[z, y, x] - 1]
            self.grid[z, y, x] = 0
            self._spawn_particles(x, y, z, color)
            self.flash_effects.append({'pos': (x, y, z), 'timer': 5, 'color': color})

        return reward, len(to_clear), 1 # 1 group cleared

    def _settle_grid(self):
        for x in range(self.GRID_W):
            for y in range(self.GRID_D):
                write_idx = 0
                for z in range(self.GRID_H):
                    if self.grid[z, y, x] != 0:
                        if z != write_idx:
                            self.grid[write_idx, y, x] = self.grid[z, y, x]
                            self.grid[z, y, x] = 0
                        write_idx += 1

    def _spawn_piece(self):
        self.piece = self.next_piece
        if self.piece:
            self.piece['pos'] = [self.GRID_W // 2, self.GRID_D // 2, self.GRID_H - 2]
        
        self.next_piece = {
            'pos': [self.GRID_W + 2, self.GRID_D // 2, self.GRID_H - 4],
            'rot': self.np_random.integers(0, 4),
            'colors': (self.np_random.integers(1, len(self.JEWEL_COLORS) + 1), 
                       self.np_random.integers(1, len(self.JEWEL_COLORS) + 1))
        }
        self.gravity_timer = 0
        
    def _get_piece_coords(self, piece=None):
        if piece is None:
            piece = self.piece
        
        if piece is None:
            return []
            
        x, y, z = piece['pos']
        rot = piece['rot']
        
        p1 = [x, y, z]
        p2 = list(p1) # copy
        
        if rot == 0: p2[2] += 1 # Up
        elif rot == 1: p2[0] += 1 # Right
        elif rot == 2: p2[2] -= 1 # Down
        elif rot == 3: p2[0] -= 1 # Left

        return [tuple(p1), tuple(p2)]

    def _check_collision(self, dx, dy, dz, d_rot):
        if self.piece is None: return True

        temp_piece = {
            'pos': [self.piece['pos'][0] + dx, self.piece['pos'][1] + dy, self.piece['pos'][2] + dz],
            'rot': (self.piece['rot'] + d_rot) % 4
        }
        
        for x, y, z in self._get_piece_coords(temp_piece):
            if not (0 <= x < self.GRID_W and 0 <= y < self.GRID_D):
                return True # Wall collision (x,y)
            if not (0 <= z < self.GRID_H):
                return True # Floor/ceiling collision
            if self.grid[z, y, x] != 0:
                return True # Collision with static jewel
        return False
    
    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid_base()

        # Draw static jewels from back to front
        jewels_to_draw = []
        for z in range(self.GRID_H):
            for y in range(self.GRID_D):
                for x in range(self.GRID_W):
                    if self.grid[z, y, x] != 0:
                        color_idx = self.grid[z, y, x] - 1
                        jewels_to_draw.append(((x,y,z), self.JEWEL_COLORS[color_idx]))
        
        # Sort by screen y-pos for correct occlusion
        jewels_to_draw.sort(key=lambda item: (item[0][1] + item[0][0]), reverse=True)
        for pos, color in jewels_to_draw:
             self._draw_iso_cube(self.screen, pos[0], pos[1], pos[2], color)

        # Draw ghost piece
        self._draw_ghost_piece()

        # Draw current and next pieces
        if self.piece:
            self._draw_piece(self.piece)
        if self.next_piece:
            self._draw_piece(self.next_piece)
        
        # Draw flash effects on top
        for flash in self.flash_effects:
             self._draw_iso_cube(self.screen, flash['pos'][0], flash['pos'][1], flash['pos'][2], (255,255,255), 1.2)

        self._draw_particles()

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        cleared_text = f"Cleared: {self.match_groups_cleared} / {self.WIN_CONDITION}"

        self._draw_text(score_text, (15, 15), self.font_large)
        self._draw_text(cleared_text, (15, 50), self.font_small)
        
        self._draw_text("Next:", (self.WIDTH - 120, self.HEIGHT - 170), self.font_small)

        if self.game_over:
            if self.match_groups_cleared >= self.WIN_CONDITION:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            self._draw_text(end_text, (self.WIDTH / 2, self.HEIGHT / 2), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=None, shadow_color=None, center=False):
        if color is None: color = self.COLOR_UI_TEXT
        if shadow_color is None: shadow_color = self.COLOR_UI_SHADOW
        
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
            
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _to_screen_coords(self, x, y, z):
        sx = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        sy = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF - z * self.TILE_Z_HEIGHT
        return int(sx), int(sy)

    def _draw_iso_cube(self, surface, x, y, z, color, scale=1.0):
        # Darken colors for sides
        top_color = color
        side_color1 = tuple(max(0, c - 40) for c in color)
        side_color2 = tuple(max(0, c - 60) for c in color)

        w, h, zh = self.TILE_W_HALF * scale, self.TILE_H_HALF * scale, self.TILE_Z_HEIGHT * scale
        
        sx, sy = self._to_screen_coords(x, y, z)
        
        # Points for the cube
        points = [
            (sx, sy - zh), (sx + w, sy - zh + h), (sx, sy - zh + 2 * h), (sx - w, sy - zh + h), # Top face
            (sx, sy), (sx + w, sy + h), (sx, sy + 2*h), (sx - w, sy + h) # Bottom face (for sides)
        ]
        
        # Draw faces
        pygame.gfxdraw.filled_polygon(surface, [points[1], points[2], points[6], points[5]], side_color1)
        pygame.gfxdraw.aapolygon(surface, [points[1], points[2], points[6], points[5]], side_color1)

        pygame.gfxdraw.filled_polygon(surface, [points[2], points[3], points[7], points[6]], side_color2)
        pygame.gfxdraw.aapolygon(surface, [points[2], points[3], points[7], points[6]], side_color2)

        pygame.gfxdraw.filled_polygon(surface, [points[0], points[1], points[2], points[3]], top_color)
        pygame.gfxdraw.aapolygon(surface, [points[0], points[1], points[2], points[3]], top_color)

    def _draw_grid_base(self):
        for y in range(self.GRID_D + 1):
            start = self._to_screen_coords(0, y, 0)
            end = self._to_screen_coords(self.GRID_W, y, 0)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_W + 1):
            start = self._to_screen_coords(x, 0, 0)
            end = self._to_screen_coords(x, self.GRID_D, 0)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _draw_piece(self, piece):
        coords = self._get_piece_coords(piece)
        if not coords: return
        c1 = self.JEWEL_COLORS[piece['colors'][0] - 1]
        c2 = self.JEWEL_COLORS[piece['colors'][1] - 1]
        self._draw_iso_cube(self.screen, coords[0][0], coords[0][1], coords[0][2], c1)
        self._draw_iso_cube(self.screen, coords[1][0], coords[1][1], coords[1][2], c2)

    def _draw_ghost_piece(self):
        if not self.piece: return
        
        ghost_piece = {
            'pos': list(self.piece['pos']),
            'rot': self.piece['rot']
        }
        
        # Find landing spot
        temp_piece_ref = self.piece
        self.piece = ghost_piece
        while not self._check_collision(0, 0, -1, 0):
            ghost_piece['pos'][2] -= 1
        self.piece = temp_piece_ref
        
        coords = self._get_piece_coords(ghost_piece)
        c1 = self.JEWEL_COLORS[self.piece['colors'][0] - 1]
        c2 = self.JEWEL_COLORS[self.piece['colors'][1] - 1]
        
        for pos, color in [(coords[0], c1), (coords[1], c2)]:
            sx, sy = self._to_screen_coords(pos[0], pos[1], pos[2])
            w, h, zh = self.TILE_W_HALF, self.TILE_H_HALF, self.TILE_Z_HEIGHT
            points = [
                (sx, sy - zh), (sx + w, sy - zh + h), (sx, sy - zh + 2*h), (sx - w, sy - zh + h)
            ]
            pygame.draw.polygon(self.screen, color, points, 2)
    
    # --- Effects ---

    def _spawn_particles(self, x, y, z, color):
        sx, sy = self._to_screen_coords(x, y, z)
        for _ in range(10):
            self.particles.append({
                'pos': [sx, sy],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0)],
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_effects(self):
        # Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Flash
        for f in self.flash_effects:
            f['timer'] -= 1
        self.flash_effects = [f for f in self.flash_effects if f['timer'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = p['color']
            size = int(p['size'] * (p['life'] / 20))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (size, size), size)
                self.screen.blit(s, (p['pos'][0] - size, p['pos'][1] - size))

    # --- Gymnasium Interface ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cleared": self.match_groups_cleared
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Human Playable Demo ---
    # This __main__ block will not work correctly with SDL_VIDEODRIVER="dummy"
    # To play, you must comment out the line: os.environ['SDL_VIDEODRIVER'] = 'dummy'
    # in the __init__ method.
    
    # For automated testing, the dummy driver is required.
    try:
        # Check if we are in a headless environment
        is_headless = os.environ.get('SDL_VIDEODRIVER') == 'dummy'
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        if is_headless:
            print("Running in headless mode. Cannot create display for human play.")
            print("To play, comment out 'os.environ['SDL_VIDEODRIVER'] = 'dummy'' in __init__.")
            # Run a few random steps to ensure it works
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            print("Headless test step completed.")
            exit()

        running = True
        total_reward = 0
        
        pygame.display.set_caption("Jewel Fall")
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        action = env.action_space.sample()
        action.fill(0)

        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

            # --- Action Mapping ---
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([movement, space_held, shift_held])

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Rendering ---
            # The observation is already the rendered screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30)

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
                total_reward = 0
                pygame.time.wait(2000) # Pause before reset

        env.close()

    except pygame.error as e:
        print("\nPygame display error. This is expected if you are running in a headless environment.")
        print("To play the game, you need a display. Make sure you are not in a container or SSH session without X forwarding.")
        print(f"Original error: {e}")