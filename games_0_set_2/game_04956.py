
# Generated: 2025-08-28T03:32:58.563433
# Source Brief: brief_04956.md
# Brief Index: 4956

        
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
    """
    An expert implementation of a grid-based roguelike dungeon crawler as a Gymnasium environment.
    This environment prioritizes visual polish and engaging gameplay, featuring procedural room generation,
    turn-based combat, and a clear reward structure, all rendered with a retro pixel-art aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold space to attack in the direction you are facing."
    )

    game_description = (
        "Explore a procedural dungeon, room by room. Collect gold, battle monsters, and try to "
        "survive to reach the final room. Your health is limited, so plan your moves carefully!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_SIZE = 40
    MAX_STEPS = 1000
    NUM_ROOMS = 5

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (60, 60, 80)
    COLOR_FLOOR = (40, 40, 55)
    COLOR_DOOR = (120, 80, 50)
    
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_DMG = (255, 100, 100)
    
    COLOR_ENEMY = (200, 50, 50)
    
    COLOR_GOLD = (255, 220, 0)
    COLOR_TRAP = (180, 0, 255)
    
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables are initialized in reset()
        self.dungeon = []
        self.current_room_index = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rng = None

        # Player state
        self.player_pos = (0, 0)
        self.player_health = 0
        self.player_max_health = 10
        self.player_facing_dir = (1, 0)
        self.player_took_damage_timer = 0

        # Room state
        self.room_layout = None
        self.enemies = []
        self.gold_items = []
        self.exit_pos = (0, 0)
        self.start_pos = (0, 0)
        
        # Visual Effects
        self.vfx = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.player_max_health
        self.player_facing_dir = (1, 0)
        self.current_room_index = 0
        self.vfx.clear()

        self._generate_dungeon()
        self._load_room(0)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Action Phase ---
        did_player_act = movement != 0 or space_held
        player_moved_to_new_room = False

        if did_player_act:
            # 1. Handle Movement
            if movement != 0:
                reward += self._handle_player_movement(movement)
                if self.player_pos == self.exit_pos:
                    player_moved_to_new_room = True
            
            # 2. Handle Attack
            elif space_held:
                reward += self._handle_player_attack()

        # --- Post-Action Checks ---
        if player_moved_to_new_room:
            self.current_room_index += 1
            if self.current_room_index >= self.NUM_ROOMS:
                # VICTORY
                reward += 100
                self.game_over = True
                terminated = True
                self._add_vfx('text', (self.GRID_WIDTH / 2, self.GRID_HEIGHT / 2), text="VICTORY!", color=(255,255,100), life=60, size='large')
            else:
                # sound: door_open
                self._load_room(self.current_room_index)
        
        # --- Enemy Action Phase (only if player acted and is not dead/won) ---
        if did_player_act and not self.game_over and not player_moved_to_new_room:
            self._handle_enemy_turns()

        # --- State Update and Termination Check ---
        self.steps += 1
        self._update_vfx()
        self.player_took_damage_timer = max(0, self.player_took_damage_timer - 1)

        if self.player_health <= 0 and not self.game_over:
            # DEFEAT
            reward = -100
            self.game_over = True
            terminated = True
            self._add_vfx('text', self.player_pos, text="DEFEATED", color=(255,50,50), life=60, size='large')
            # sound: player_death
        
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Action Handlers ---
    def _handle_player_movement(self, movement_action):
        reward = 0
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        dx, dy = move_map[movement_action]
        
        self.player_facing_dir = (dx, dy)
        
        current_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        target_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        
        if not self._is_walkable(target_pos):
            # sound: bump_wall
            return 0 # Cannot move into walls or other entities

        new_dist = self._manhattan_distance(target_pos, self.exit_pos)
        reward += 0.1 if new_dist < current_dist else -0.1
        
        self.player_pos = target_pos
        # sound: player_step
        
        # Check for gold/traps
        item_idx = self._get_item_at(self.player_pos)
        if item_idx is not None:
            item = self.gold_items.pop(item_idx)
            if item['is_trap']:
                # sound: trap_triggered
                self.player_health -= 2
                self.player_took_damage_timer = 5
                reward -= 2
                self._add_vfx('text', self.player_pos, text="-2 HP", color=(255, 100, 255), life=20)
            else:
                # sound: gold_pickup
                self.score += 10
                reward += 1
                self._add_vfx('text', self.player_pos, text="+10", color=self.COLOR_GOLD, life=20)
        return reward

    def _handle_player_attack(self):
        reward = 0
        attack_pos = (self.player_pos[0] + self.player_facing_dir[0], self.player_pos[1] + self.player_facing_dir[1])
        
        self._add_vfx('slash', attack_pos, direction=self.player_facing_dir, life=5)
        # sound: player_attack_swing

        enemy_idx = self._get_enemy_at(attack_pos)
        if enemy_idx is not None:
            enemy = self.enemies[enemy_idx]
            player_damage = 2
            enemy['health'] -= player_damage
            # sound: player_attack_hit
            self._add_vfx('text', enemy['pos'], text=f"-{player_damage}", color=(255, 255, 255), life=15)
            
            if enemy['health'] <= 0:
                self.enemies.pop(enemy_idx)
                reward += 1
                # sound: enemy_death
        return reward

    def _handle_enemy_turns(self):
        for enemy in self.enemies:
            if self._manhattan_distance(self.player_pos, enemy['pos']) == 1:
                # Attack player
                enemy_damage = 1
                self.player_health -= enemy_damage
                self.player_took_damage_timer = 5
                # sound: player_hit
                self._add_vfx('text', self.player_pos, text=f"-{enemy_damage}", color=self.COLOR_PLAYER_DMG, life=20)
            else:
                # Move towards player
                dx = self.player_pos[0] - enemy['pos'][0]
                dy = self.player_pos[1] - enemy['pos'][1]
                
                move_options = []
                if dx != 0: move_options.append((np.sign(dx), 0))
                if dy != 0: move_options.append((0, np.sign(dy)))
                
                self.rng.shuffle(move_options)

                for move_dx, move_dy in move_options:
                    target_pos = (enemy['pos'][0] + move_dx, enemy['pos'][1] + move_dy)
                    if self._is_walkable(target_pos, for_enemy=True):
                        enemy['pos'] = target_pos
                        break

    # --- Generation and Loading ---
    def _generate_dungeon(self):
        self.dungeon = []
        for i in range(self.NUM_ROOMS):
            layout = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int) # 1 = wall
            
            start_y = self.rng.integers(1, self.GRID_HEIGHT - 1)
            exit_y = self.rng.integers(1, self.GRID_HEIGHT - 1)
            start_pos = (0, start_y)
            exit_pos = (self.GRID_WIDTH - 1, exit_y)

            # Carve path
            px, py = start_pos
            layout[px, py] = 0
            while (px, py) != exit_pos:
                move_x = px < exit_pos[0]
                move_y = py != exit_pos[1]
                
                if move_x and (self.rng.random() > 0.4 or not move_y):
                    px += 1
                elif move_y:
                    py += np.sign(exit_pos[1] - py)
                layout[px, py] = 0

            # Add some randomness and open space
            floor_tiles = list(zip(*np.where(layout == 0)))
            for _ in range(50):
                if not floor_tiles: break
                cx, cy = self.rng.choice(floor_tiles)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = cx + dx, cy + dy
                        if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1:
                            if self.rng.random() > 0.4:
                                layout[nx, ny] = 0

            floor_tiles = list(zip(*np.where(layout == 0)))
            
            # Place enemies (not in first room)
            enemies = []
            if i > 0:
                num_enemies = self.rng.integers(1, 3)
                enemy_health = 5 + (i - 1)
                for _ in range(num_enemies):
                    if len(floor_tiles) > 5:
                        pos = self.rng.choice(floor_tiles)
                        if self._manhattan_distance(pos, start_pos) > 3:
                            enemies.append({'pos': tuple(pos), 'health': enemy_health, 'max_health': enemy_health})
                            floor_tiles.remove(tuple(pos))
            
            # Place gold/traps
            gold_items = []
            num_gold = self.rng.integers(2, 5)
            for _ in range(num_gold):
                if floor_tiles:
                    pos = self.rng.choice(floor_tiles)
                    is_trap = self.rng.random() < 0.1
                    gold_items.append({'pos': tuple(pos), 'is_trap': is_trap})
                    floor_tiles.remove(tuple(pos))

            self.dungeon.append({
                'layout': layout,
                'start_pos': start_pos,
                'exit_pos': exit_pos,
                'enemies': enemies,
                'gold_items': gold_items
            })

    def _load_room(self, room_index):
        room_data = self.dungeon[room_index]
        self.room_layout = room_data['layout']
        self.start_pos = room_data['start_pos']
        self.exit_pos = room_data['exit_pos']
        self.player_pos = self.start_pos
        self.enemies = [e.copy() for e in room_data['enemies']]
        self.gold_items = [g.copy() for g in room_data['gold_items']]
        self.vfx.clear()

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_vfx()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.room_layout[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        pygame.draw.rect(self.screen, self.COLOR_DOOR, (self.exit_pos[0] * self.TILE_SIZE, self.exit_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        for item in self.gold_items:
            cx, cy = (int((item['pos'][0] + 0.5) * self.TILE_SIZE), int((item['pos'][1] + 0.5) * self.TILE_SIZE))
            if item['is_trap']:
                p1 = (cx, cy - 8)
                p2 = (cx - 8, cy + 8)
                p3 = (cx + 8, cy + 8)
                pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_TRAP)
                pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_TRAP)
            else:
                pygame.gfxdraw.aacircle(self.screen, cx, cy, 8, self.COLOR_GOLD)
                pygame.gfxdraw.filled_circle(self.screen, cx, cy, 8, self.COLOR_GOLD)

        for enemy in self.enemies:
            rect = (enemy['pos'][0] * self.TILE_SIZE + 4, enemy['pos'][1] * self.TILE_SIZE + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)
            # Enemy health bar
            if enemy['health'] < enemy['max_health']:
                hp_ratio = enemy['health'] / enemy['max_health']
                bar_w = self.TILE_SIZE - 8
                hp_bar_rect = (rect[0], rect[1] - 8, bar_w * hp_ratio, 5)
                hp_bar_bg_rect = (rect[0], rect[1] - 8, bar_w, 5)
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, hp_bar_bg_rect)
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, hp_bar_rect)

        player_color = self.COLOR_PLAYER_DMG if self.player_took_damage_timer > 0 else self.COLOR_PLAYER
        rect = (self.player_pos[0] * self.TILE_SIZE + 2, self.player_pos[1] * self.TILE_SIZE + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
        pygame.draw.rect(self.screen, player_color, rect, border_radius=4)
        
        # Player facing indicator
        center_x = rect[0] + rect[2] / 2
        center_y = rect[1] + rect[3] / 2
        eye_x = center_x + self.player_facing_dir[0] * 8
        eye_y = center_y + self.player_facing_dir[1] * 8
        pygame.draw.circle(self.screen, (255,255,255), (int(eye_x), int(eye_y)), 3)

    def _render_ui(self):
        # Health Bar
        hp_ratio = max(0, self.player_health / self.player_max_health)
        hp_bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, hp_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, hp_bar_width * hp_ratio, 20))
        hp_text = self.font_small.render(f"HP: {self.player_health}/{self.player_max_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hp_text, (15, 12))

        # Score
        score_text = self.font_large.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Room Number
        room_text = self.font_large.render(f"Room {self.current_room_index + 1} / {self.NUM_ROOMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(room_text, (self.SCREEN_WIDTH / 2 - room_text.get_width() / 2, self.SCREEN_HEIGHT - 30))

    def _render_vfx(self):
        for vfx in self.vfx:
            pos_x, pos_y = (vfx['pos'][0] + 0.5) * self.TILE_SIZE, (vfx['pos'][1] + 0.5) * self.TILE_SIZE
            
            if vfx['type'] == 'text':
                font = self.font_large if vfx.get('size') == 'large' else self.font_small
                alpha = int(255 * (vfx['life'] / vfx['max_life']))
                text_surf = font.render(vfx['text'], True, vfx['color'])
                text_surf.set_alpha(alpha)
                pos_y -= (vfx['max_life'] - vfx['life']) # Float up
                self.screen.blit(text_surf, (pos_x - text_surf.get_width() / 2, pos_y - text_surf.get_height() / 2))
            
            elif vfx['type'] == 'slash':
                alpha = int(255 * (vfx['life'] / vfx['max_life']))
                color = (*self.COLOR_UI_TEXT, alpha)
                dx, dy = vfx['direction']
                line_len = 20 * (1 - (vfx['life'] / vfx['max_life']))
                if dx != 0: # Horizontal slash
                    start = (pos_x - line_len/2 * dx, pos_y - 10)
                    end = (pos_x + line_len/2 * dx, pos_y + 10)
                else: # Vertical slash
                    start = (pos_x - 10, pos_y - line_len/2 * dy)
                    end = (pos_x + 10, pos_y + line_len/2 * dy)
                pygame.draw.line(self.screen, color, start, end, 3)

    # --- Helpers ---
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _is_walkable(self, pos, for_enemy=False):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT): return False
        if self.room_layout[x, y] == 1: return False
        if not for_enemy and pos == self.player_pos: return False
        if self._get_enemy_at(pos) is not None: return False
        return True

    def _get_enemy_at(self, pos):
        for i, enemy in enumerate(self.enemies):
            if enemy['pos'] == pos:
                return i
        return None

    def _get_item_at(self, pos):
        for i, item in enumerate(self.gold_items):
            if item['pos'] == pos:
                return i
        return None

    def _add_vfx(self, vfx_type, pos, **kwargs):
        vfx_data = {'type': vfx_type, 'pos': pos, 'life': kwargs.get('life', 20)}
        vfx_data['max_life'] = vfx_data['life']
        vfx_data.update(kwargs)
        self.vfx.append(vfx_data)

    def _update_vfx(self):
        self.vfx = [v for v in self.vfx if v['life'] > 0]
        for v in self.vfx:
            v['life'] -= 1

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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while not terminated:
        movement = 0 # no-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # Movement, Space, Shift
        
        if movement != 0 or space == 1:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # In manual play, we add a small delay to make turns visible
        if movement != 0 or space == 1:
            pygame.time.wait(100) 

        clock.tick(30) # Limit FPS
        
    print("Game Over!")
    env.close()