import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend the ancient trees by summoning elemental guardians. Select lettered tiles to form "
        "recipes and unleash your defenders against waves of encroaching blight."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to select letter tiles to form "
        "recipes. Press shift to end your turn and trigger the action phase."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GAME_AREA_WIDTH = 480
    UI_WIDTH = SCREEN_WIDTH - GAME_AREA_WIDTH
    GRID_SIZE = 4
    TILE_SIZE = 40
    GRID_MARGIN_X = (GAME_AREA_WIDTH - GRID_SIZE * TILE_SIZE) // 2
    GRID_MARGIN_Y = 280

    # Colors
    COLOR_BG = (12, 20, 33)
    COLOR_TREE_HEALTHY = (40, 180, 99)
    COLOR_TREE_TRUNK = (87, 65, 47)
    COLOR_BLIGHT = (192, 57, 43)
    COLOR_CURSOR = (241, 196, 15, 200)
    COLOR_TEXT = (236, 240, 241)
    COLOR_TEXT_DIM = (149, 165, 166)
    COLOR_GRID_BG = (44, 62, 80, 150)
    COLOR_TILE_BG = (52, 73, 94)
    COLOR_TILE_SELECTED_BG = (22, 160, 133)
    
    GUARDIAN_COLORS = {
        "WATER": (52, 152, 219),
        "FIRE": (230, 126, 34),
        "EARTH": (149, 115, 80),
        "LIGHT": (241, 196, 15),
    }

    GUARDIAN_RECIPES = {
        "AB": "WATER",
        "CD": "FIRE",
        "BCA": "EARTH",
        "DDAA": "LIGHT",
    }
    
    MAX_STEPS = 1000
    WIN_WAVE = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_small = pygame.font.SysFont("Consolas", 14)
            self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 16)
            self.font_medium = pygame.font.Font(None, 22)
            self.font_large = pygame.font.Font(None, 30)

        # These attributes are defined here and initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.trees = []
        self.blight_units = []
        self.guardians = []
        self.particles = []
        self.tile_grid = np.array([])
        self.cursor_pos = [0, 0]
        self.selected_tiles_data = []
        self.unlocked_guardians = set()
        self.last_space_held = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        
        self.trees = [
            {'pos': (40, 150), 'hp': 100, 'max_hp': 100, 'radius': 20},
            {'pos': (80, 80), 'hp': 100, 'max_hp': 100, 'radius': 20},
            {'pos': (80, 220), 'hp': 100, 'max_hp': 100, 'radius': 20},
        ]
        
        self.blight_units = []
        self.guardians = []
        self.particles = []
        
        self.cursor_pos = [0, 0]
        self.selected_tiles_data = []
        self.unlocked_guardians = {"WATER", "FIRE", "EARTH"}
        self.last_space_held = False

        self._generate_tiles()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.01  # Small penalty for taking time
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_player_input(movement, space_held)
        
        # Shift triggers the end of the player's turn and the start of the action phase
        if shift_held:
            turn_reward = self._execute_turn()
            reward += turn_reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        # Apply terminal rewards
        if (terminated or truncated) and not self.game_over:
            total_hp = sum(t['hp'] for t in self.trees)
            if self.wave > self.WIN_WAVE:
                reward += 100
                self.score += 100
            elif total_hp <= 0:
                reward -= 100
                self.score -= 100
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # Select tile on space press (rising edge)
        if space_held and not self.last_space_held:
            grid_x, grid_y = self.cursor_pos
            tile_char = self.tile_grid[grid_y, grid_x]
            
            # Prevent selecting the same tile twice
            is_already_selected = any(d['grid_pos'] == (grid_x, grid_y) for d in self.selected_tiles_data)
            
            if not is_already_selected:
                tile_world_pos = (
                    self.GRID_MARGIN_X + grid_x * self.TILE_SIZE + self.TILE_SIZE // 2,
                    self.GRID_MARGIN_Y + grid_y * self.TILE_SIZE + self.TILE_SIZE // 2
                )
                self.selected_tiles_data.append({
                    'char': tile_char,
                    'world_pos': tile_world_pos,
                    'grid_pos': (grid_x, grid_y)
                })
                self._create_particles(tile_world_pos, self.COLOR_TILE_SELECTED_BG, 5)

        self.last_space_held = space_held

    def _execute_turn(self):
        reward = 0
        
        # 1. Summon Guardian
        word = "".join([d['char'] for d in self.selected_tiles_data])
        if word and word in self.GUARDIAN_RECIPES:
            g_type = self.GUARDIAN_RECIPES[word]
            if g_type in self.unlocked_guardians:
                summon_pos = self.selected_tiles_data[-1]['world_pos']
                self._summon_guardian(g_type, summon_pos)
        
        self.selected_tiles_data.clear()

        # 2. Guardians Attack
        for guardian in self.guardians:
            reward += self._guardian_ai(guardian)

        # 3. Blight Move & Attack
        blight_to_remove = []
        for blight in self.blight_units:
            blight['pos'] = (blight['pos'][0] - blight['speed'], blight['pos'][1])
            hit_tree = False
            for i, tree in enumerate(self.trees):
                if tree['hp'] > 0 and math.hypot(blight['pos'][0] - tree['pos'][0], blight['pos'][1] - tree['pos'][1]) < blight['radius'] + tree['radius']:
                    damage = 10
                    tree['hp'] = max(0, tree['hp'] - damage)
                    reward -= 0.5 * damage
                    self._create_particles(tree['pos'], self.COLOR_BLIGHT, 15, life=30)
                    blight_to_remove.append(blight)
                    hit_tree = True
                    break
            if not hit_tree and blight['pos'][0] < 0:
                blight_to_remove.append(blight)

        self.blight_units = [b for b in self.blight_units if b not in blight_to_remove]
        
        # 4. Check for Wave End
        if not self.blight_units:
            reward += 2.0
            self.score += 2
            self.wave += 1
            if self.wave == 6:
                self.unlocked_guardians.add("LIGHT")
            if self.wave <= self.WIN_WAVE:
                self._spawn_wave()
        
        # 5. Replenish Tiles
        self._generate_tiles()

        return reward

    def _guardian_ai(self, guardian):
        reward = 0
        guardian['cooldown'] = max(0, guardian['cooldown'] - 1)
        if guardian['cooldown'] > 0:
            return 0

        target = None
        min_dist = float('inf')
        for blight in self.blight_units:
            dist = math.hypot(guardian['pos'][0] - blight['pos'][0], guardian['pos'][1] - blight['pos'][1])
            if dist < guardian['range'] and dist < min_dist:
                min_dist = dist
                target = blight

        if target:
            guardian['cooldown'] = guardian['rate']
            self._create_attack_particle(guardian, target)
            
            target['hp'] -= guardian['damage']
            if target['hp'] <= 0:
                self.blight_units.remove(target)
                reward += 0.1
                self.score += 0.1
                self._create_particles(target['pos'], self.COLOR_BLIGHT, 30, life=40)
        return reward

    def _summon_guardian(self, g_type, pos):
        stats = {
            "WATER": {'range': 100, 'damage': 5, 'rate': 30, 'radius': 12},
            "FIRE": {'range': 80, 'damage': 15, 'rate': 60, 'radius': 12},
            "EARTH": {'range': 150, 'damage': 10, 'rate': 45, 'radius': 12},
            "LIGHT": {'range': 200, 'damage': 25, 'rate': 75, 'radius': 12},
        }[g_type]
        
        self.guardians.append({
            'type': g_type,
            'pos': pos,
            'cooldown': 0,
            'anim_offset': self.np_random.uniform(0, 2 * math.pi),
            **stats
        })
        self._create_particles(pos, self.GUARDIAN_COLORS[g_type], 20, life=40)

    def _generate_tiles(self):
        self.tile_grid = self.np_random.choice(['A', 'B', 'C', 'D'], size=(self.GRID_SIZE, self.GRID_SIZE))

    def _spawn_wave(self):
        blight_count = 3 + self.wave
        base_speed = 0.5 + (self.wave // 2) * 0.1
        base_hp = 10 + (self.wave // 5) * 5
        
        for _ in range(blight_count):
            self.blight_units.append({
                'pos': (self.GAME_AREA_WIDTH + self.np_random.integers(20, 101), 
                        self.np_random.integers(40, self.SCREEN_HEIGHT - 40 + 1)),
                'hp': base_hp,
                'max_hp': base_hp,
                'speed': base_speed + self.np_random.uniform(-0.1, 0.1),
                'radius': 8,
                'anim_offset': self.np_random.uniform(0, 2 * math.pi)
            })

    def _check_termination(self):
        total_hp = sum(t['hp'] for t in self.trees)
        return total_hp <= 0 or self.wave > self.WIN_WAVE

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "tree_health": [t['hp'] for t in self.trees],
            "guardians": len(self.guardians),
            "blight": len(self.blight_units),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_trees()
        self._render_guardians()
        self._render_blight()
        self._render_tile_grid()
        self._update_and_render_particles()

    def _render_trees(self):
        for tree in self.trees:
            if tree['hp'] > 0:
                # Trunk
                pygame.draw.rect(self.screen, self.COLOR_TREE_TRUNK, (tree['pos'][0] - 5, tree['pos'][1], 10, 20))
                # Foliage
                self._draw_glowing_circle(self.screen, tree['pos'], tree['radius'], self.COLOR_TREE_HEALTHY)
                # Health bar
                health_ratio = tree['hp'] / tree['max_hp']
                bar_width = tree['radius'] * 2
                pygame.draw.rect(self.screen, self.COLOR_BLIGHT, (tree['pos'][0] - bar_width//2, tree['pos'][1] - tree['radius'] - 10, bar_width, 5))
                pygame.draw.rect(self.screen, self.COLOR_TREE_HEALTHY, (tree['pos'][0] - bar_width//2, tree['pos'][1] - tree['radius'] - 10, int(bar_width * health_ratio), 5))

    def _render_guardians(self):
        for g in self.guardians:
            anim_y = math.sin(self.steps * 0.1 + g['anim_offset']) * 3
            pos = (int(g['pos'][0]), int(g['pos'][1] + anim_y))
            color = self.GUARDIAN_COLORS[g['type']]
            self._draw_glowing_circle(self.screen, pos, g['radius'], color)

    def _render_blight(self):
        for b in self.blight_units:
            anim_y = math.sin(self.steps * 0.2 + b['anim_offset']) * 2
            pos = (int(b['pos'][0]), int(b['pos'][1] + anim_y))
            pygame.draw.rect(self.screen, self.COLOR_BLIGHT, (pos[0] - b['radius'], pos[1] - b['radius'], b['radius']*2, b['radius']*2))

    def _render_tile_grid(self):
        # Draw selected path
        if len(self.selected_tiles_data) > 1:
            points = [d['world_pos'] for d in self.selected_tiles_data]
            pygame.draw.lines(self.screen, self.COLOR_TILE_SELECTED_BG, False, points, 3)

        # Draw grid and tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(self.GRID_MARGIN_X + c * self.TILE_SIZE, self.GRID_MARGIN_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                
                is_selected = any(d['grid_pos'] == (c, r) for d in self.selected_tiles_data)
                bg_color = self.COLOR_TILE_SELECTED_BG if is_selected else self.COLOR_TILE_BG
                pygame.draw.rect(self.screen, bg_color, rect, border_radius=4)
                
                char = self.tile_grid[r, c]
                text_surf = self.font_large.render(char, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cursor_rect = pygame.Rect(self.GRID_MARGIN_X + self.cursor_pos[0] * self.TILE_SIZE, self.GRID_MARGIN_Y + self.cursor_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

    def _render_ui(self):
        ui_bg = pygame.Surface((self.UI_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        ui_bg.fill((20, 30, 45, 200))
        self.screen.blit(ui_bg, (self.GAME_AREA_WIDTH, 0))
        
        y_offset = 20
        # Wave
        self._draw_text(f"Wave: {self.wave}/{self.WIN_WAVE}", (self.GAME_AREA_WIDTH + 10, y_offset), self.font_medium, self.COLOR_TEXT)
        y_offset += 30
        # Score
        self._draw_text(f"Score: {self.score:.1f}", (self.GAME_AREA_WIDTH + 10, y_offset), self.font_medium, self.COLOR_TEXT)
        y_offset += 40

        # Selected Word
        self._draw_text("Selected:", (self.GAME_AREA_WIDTH + 10, y_offset), self.font_medium, self.COLOR_TEXT_DIM)
        y_offset += 25
        word = "".join([d['char'] for d in self.selected_tiles_data])
        self._draw_text(f"> {word}", (self.GAME_AREA_WIDTH + 10, y_offset), self.font_large, self.COLOR_TEXT)
        y_offset += 40

        # Recipes
        self._draw_text("Recipes:", (self.GAME_AREA_WIDTH + 10, y_offset), self.font_medium, self.COLOR_TEXT_DIM)
        y_offset += 25
        for recipe, g_type in self.GUARDIAN_RECIPES.items():
            is_unlocked = g_type in self.unlocked_guardians
            color = self.GUARDIAN_COLORS[g_type] if is_unlocked else self.COLOR_TEXT_DIM
            self._draw_text(f"{recipe}: {g_type}", (self.GAME_AREA_WIDTH + 20, y_offset), self.font_small, color)
            y_offset += 20

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_glowing_circle(self, surface, pos, radius, color):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(4):
            alpha = 150 - i * 35
            if alpha <= 0: continue
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius + i * 2), (*color, alpha))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)

    def _create_particles(self, pos, color, count, life=20, speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = speed * self.np_random.uniform(0.5, 1.5)
            vel = (math.cos(angle) * vel_mag, math.sin(angle) * vel_mag)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _create_attack_particle(self, start_entity, end_entity):
        self.particles.append({
            'pos': list(start_entity['pos']),
            'target_pos': end_entity['pos'],
            'life': 15, 'max_life': 15,
            'color': self.GUARDIAN_COLORS[start_entity['type']],
            'type': 'beam'
        })

    def _update_and_render_particles(self):
        for p in reversed(self.particles):
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            life_ratio = p['life'] / p['max_life']
            color = (*p['color'], int(255 * life_ratio))
            
            if p.get('type') == 'beam':
                # Interpolate position for beam
                progress = 1.0 - life_ratio
                start_pos = p['pos']
                end_pos = p['target_pos']
                current_pos = (
                    start_pos[0] + (end_pos[0] - start_pos[0]) * progress,
                    start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                )
                pygame.draw.circle(self.screen, color, (int(current_pos[0]), int(current_pos[1])), 4)
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95) # friction
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, color, pos, int(3 * life_ratio))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will create a window and capture keyboard input.
    # The environment itself remains headless and is not affected.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Ancient Guardian Defense")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # [movement, space, shift]

    print("--- Manual Play Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    while not done:
        # Process keyboard events for manual control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action for this frame
        action = [0, 0, 0]

        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) 

    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")