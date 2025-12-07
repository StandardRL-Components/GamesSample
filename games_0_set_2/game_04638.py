
# Generated: 2025-08-28T03:01:28.076817
# Source Brief: brief_04638.md
# Brief Index: 4638

        
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
        "Controls: Arrows to move cursor. Space to select/deselect a crystal. "
        "Arrows to move a selected crystal. Shift to cancel selection."
    )

    game_description = (
        "Navigate a crystal cavern. Move crystals to redirect light beams and "
        "illuminate all of them before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (26, 28, 44)
    COLOR_GRID = (40, 42, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UNLIT = (80, 80, 90)
    CRYSTAL_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 0),    # Green
    ]
    
    # Grid and World
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 16, 12
    TILE_W, TILE_H = 48, 24
    ISO_OFFSET_X = SCREEN_W // 2
    ISO_OFFSET_Y = 100
    
    # Game Rules
    MAX_MOVES = 20
    MAX_STEPS = 1000
    NUM_CRYSTALS = 6
    MAX_BEAM_BOUNCES = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.animation_timer = 0
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win_state = False
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_crystal_idx = None
        
        self._generate_level()
        self.beams = []
        self._calculate_beams()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        move_made = False

        prev_lit_indices = {i for i, c in enumerate(self.crystals) if c['is_lit']}

        # --- Action Handling ---
        if shift_pressed and self.selected_crystal_idx is not None:
            # Deselect
            self.selected_crystal_idx = None
            # sfx: cancel_sound

        elif space_pressed:
            # Toggle selection
            if self.selected_crystal_idx is not None:
                self.selected_crystal_idx = None
                # sfx: deselect_sound
            else:
                for i, crystal in enumerate(self.crystals):
                    if crystal['pos'] == self.cursor_pos:
                        self.selected_crystal_idx = i
                        # sfx: select_sound
                        break
        
        elif movement != 0:
            # Move selected crystal OR move cursor
            if self.selected_crystal_idx is not None:
                # Attempt to move crystal
                crystal = self.crystals[self.selected_crystal_idx]
                target_pos = list(crystal['pos']) # Important: copy
                
                if movement == 1: target_pos[1] -= 1 # Up
                elif movement == 2: target_pos[1] += 1 # Down
                elif movement == 3: target_pos[0] -= 1 # Left
                elif movement == 4: target_pos[0] += 1 # Right

                # Check validity of move
                is_valid = True
                if not (0 <= target_pos[0] < self.GRID_W and 0 <= target_pos[1] < self.GRID_H):
                    is_valid = False
                if is_valid:
                    for i, other_crystal in enumerate(self.crystals):
                        if i != self.selected_crystal_idx and other_crystal['pos'] == target_pos:
                            is_valid = False
                            break
                
                if is_valid:
                    crystal['pos'] = target_pos
                    self.moves_left -= 1
                    move_made = True
                    self.selected_crystal_idx = None # Deselect after move
                    self.cursor_pos = list(target_pos)
                    # sfx: move_crystal_sound
                else:
                    # sfx: invalid_move_sound
                    pass

            else: # Move cursor
                if movement == 1: self.cursor_pos[1] -= 1 # Up
                elif movement == 2: self.cursor_pos[1] += 1 # Down
                elif movement == 3: self.cursor_pos[0] -= 1 # Left
                elif movement == 4: self.cursor_pos[0] += 1 # Right
                self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)
                # sfx: cursor_move_sound

        # --- State & Reward Calculation ---
        if move_made:
            reward -= 0.1
            self._calculate_beams()
            
            current_lit_indices = {i for i, c in enumerate(self.crystals) if c['is_lit']}
            newly_lit = current_lit_indices - prev_lit_indices

            if newly_lit:
                reward += len(newly_lit) * 5
                # sfx: new_crystal_lit_sound
                for i in newly_lit:
                    self._create_particles(self.crystals[i]['pos'], self.crystals[i]['color'])
        
        self.score += reward
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        all_lit = all(c['is_lit'] for c in self.crystals)
        
        if all_lit:
            reward += 50
            terminated = True
            self.game_over = True
            self.win_state = True
            # sfx: win_sound
        
        if self.moves_left <= 0 and not all_lit:
            reward -= 10
            terminated = True
            self.game_over = True
            # sfx: lose_sound
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.crystals = []
        occupied_pos = set()

        # Place the light source
        source_pos = [self.np_random.integers(1, self.GRID_W-1), self.np_random.integers(1, self.GRID_H-1)]
        source_dir_options = [[1,0], [-1,0], [0,1], [0,-1]]
        source_dir = list(self.np_random.choice(source_dir_options, 1)[0])
        
        source_crystal = {
            'pos': source_pos,
            'color': self.CRYSTAL_COLORS[0],
            'is_source': True,
            'source_dir': source_dir,
            'is_lit': True,
        }
        self.crystals.append(source_crystal)
        occupied_pos.add(tuple(source_pos))

        # Place other crystals
        for i in range(1, self.NUM_CRYSTALS):
            pos = [self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)]
            while tuple(pos) in occupied_pos:
                pos = [self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)]
            
            crystal = {
                'pos': pos,
                'color': self.CRYSTAL_COLORS[self.np_random.integers(0, len(self.CRYSTAL_COLORS))],
                'is_source': False,
                'is_lit': False,
            }
            self.crystals.append(crystal)
            occupied_pos.add(tuple(pos))

    def _calculate_beams(self):
        # Reset lit status and beams
        for c in self.crystals:
            if not c['is_source']:
                c['is_lit'] = False
        self.beams = []
        
        crystal_pos_map = {tuple(c['pos']): i for i, c in enumerate(self.crystals)}
        
        active_rays = []
        for i, c in enumerate(self.crystals):
            if c['is_source']:
                active_rays.append({
                    'start': c['pos'],
                    'dir': c['source_dir'],
                    'color': c['color'],
                    'bounces': self.MAX_BEAM_BOUNCES
                })

        processed_rays = set()

        while active_rays:
            ray = active_rays.pop(0)
            
            ray_key = (tuple(ray['start']), tuple(ray['dir']), ray['bounces'])
            if ray_key in processed_rays:
                continue
            processed_rays.add(ray_key)

            pos = list(ray['start'])
            direction = ray['dir']
            
            for _ in range(max(self.GRID_W, self.GRID_H) * 2):
                pos[0] += direction[0]
                pos[1] += direction[1]

                # Check for wall collision
                if not (0 <= pos[0] < self.GRID_W and 0 <= pos[1] < self.GRID_H):
                    pos[0] -= direction[0]
                    pos[1] -= direction[1]
                    self.beams.append({'start': ray['start'], 'end': pos, 'color': ray['color']})
                    
                    if ray['bounces'] > 0:
                        new_dir = list(direction)
                        if not (0 <= pos[0] + direction[0] < self.GRID_W): new_dir[0] *= -1
                        if not (0 <= pos[1] + direction[1] < self.GRID_H): new_dir[1] *= -1
                        
                        active_rays.append({
                            'start': pos, 'dir': new_dir, 'color': ray['color'], 'bounces': ray['bounces'] - 1
                        })
                    break

                # Check for crystal collision
                if tuple(pos) in crystal_pos_map:
                    self.beams.append({'start': ray['start'], 'end': pos, 'color': ray['color']})
                    hit_crystal_idx = crystal_pos_map[tuple(pos)]
                    self.crystals[hit_crystal_idx]['is_lit'] = True
                    break

    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_OFFSET_X + (x - y) * (self.TILE_W / 2)
        screen_y = self.ISO_OFFSET_Y + (x + y) * (self.TILE_H / 2)
        return int(screen_x), int(screen_y)

    def _render_grid(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), self.COLOR_GRID)

    def _render_cursor(self):
        if self.game_over: return
        x, y = self.cursor_pos
        p1 = self._iso_to_screen(x, y)
        p2 = self._iso_to_screen(x + 1, y)
        p3 = self._iso_to_screen(x + 1, y + 1)
        p4 = self._iso_to_screen(x, y + 1)
        
        pulse = (math.sin(self.animation_timer * 0.2) + 1) / 2
        color = tuple(int(c * (0.7 + pulse * 0.3)) for c in self.COLOR_CURSOR)
        
        pygame.draw.lines(self.screen, color, True, (p1, p2, p3, p4), 2)
        
    def _render_crystals(self):
        for i, crystal in enumerate(self.crystals):
            x, y = crystal['pos']
            base_color = crystal['color'] if crystal['is_lit'] else self.COLOR_UNLIT
            
            # Isometric cube points
            top_center = self._iso_to_screen(x + 0.5, y + 0.5)
            h, w = self.TILE_H, self.TILE_W
            
            p_top = (top_center[0], top_center[1] - h * 0.25)
            p_left = (top_center[0] - w * 0.25, top_center[1])
            p_right = (top_center[0] + w * 0.25, top_center[1])
            p_bottom = (top_center[0], top_center[1] + h * 0.25)
            
            # Glow for lit crystals
            if crystal['is_lit']:
                pulse = (math.sin(self.animation_timer * 0.1 + i) + 1) / 2
                glow_radius = int(self.TILE_W * (0.8 + pulse * 0.2))
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*base_color, 40), (glow_radius, glow_radius), glow_radius)
                pygame.draw.circle(glow_surf, (*base_color, 60), (glow_radius, glow_radius), int(glow_radius * 0.6))
                self.screen.blit(glow_surf, (top_center[0] - glow_radius, top_center[1] - glow_radius))

            # Crystal body
            # Top face
            top_face_color = tuple(min(255, c + 40) for c in base_color)
            pygame.gfxdraw.filled_polygon(self.screen, (p_top, p_left, p_bottom, p_right), top_face_color)
            pygame.gfxdraw.aapolygon(self.screen, (p_top, p_left, p_bottom, p_right), top_face_color)
            
            # Selection highlight
            if i == self.selected_crystal_idx:
                pygame.draw.polygon(self.screen, self.COLOR_CURSOR, (p_top, p_left, p_bottom, p_right), 3)

    def _render_beams(self):
        for beam in self.beams:
            start_scr = self._iso_to_screen(beam['start'][0] + 0.5, beam['start'][1] + 0.5)
            end_scr = self._iso_to_screen(beam['end'][0] + 0.5, beam['end'][1] + 0.5)
            color = beam['color']
            
            # Glow line
            pygame.draw.line(self.screen, (*color, 100), start_scr, end_scr, 5)
            # Core line
            pygame.draw.line(self.screen, (255, 255, 255), start_scr, end_scr, 2)
            
    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                size = int(p['size'] * (p['life'] / p['max_life']))
                if size > 0:
                    surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (size, size), size)
                    self.screen.blit(surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _create_particles(self, grid_pos, color):
        screen_pos = self._iso_to_screen(grid_pos[0] + 0.5, grid_pos[1] + 0.5)
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': list(screen_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'max_life': 40,
                'color': color,
                'size': random.randint(2, 5)
            })

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_m.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (10, 10))

        # Objective
        lit_count = sum(1 for c in self.crystals if c['is_lit'])
        total_count = len(self.crystals)
        obj_text = self.font_m.render(f"Illuminated: {lit_count} / {total_count}", True, (255, 255, 255))
        self.screen.blit(obj_text, (self.SCREEN_W - obj_text.get_width() - 10, 10))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "OUT OF MOVES"
            color = (0, 255, 128) if self.win_state else (255, 80, 80)
            
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_W/2, self.SCREEN_H/2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.animation_timer += 1
        
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_cursor()
        self._render_beams()
        self._render_crystals()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "illuminated": sum(1 for c in self.crystals if c['is_lit']),
            "selected_crystal": self.selected_crystal_idx,
            "cursor_pos": self.cursor_pos
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with a no-op

    print("\n" + "="*30)
    print("CRYSTAL CAVERNS")
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human Input to Action Mapping ---
        # This is a one-time event mapping, not continuous holding
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    continue

        action = np.array([movement, space, shift])

        # If any action was taken, step the environment
        if action.any():
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(
                f"Step: {info['steps']}, "
                f"Reward: {reward:.2f}, "
                f"Score: {info['score']:.2f}, "
                f"Moves: {info['moves_left']}, "
                f"Lit: {info['illuminated']}"
            )

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it from the env and display it
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    print("\n--- GAME OVER ---")
    env.close()