
# Generated: 2025-08-28T02:27:16.584255
# Source Brief: brief_01707.md
# Brief Index: 1707

        
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
        "Controls: Arrow keys to move. Space to attack. A turn passes after each action."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated isometric dungeon. Defeat enemies, collect gold, and find the exit to progress through 10 rooms."
    )

    # Frames only advance when an action is received.
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 15, 15
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    MAX_ROOMS = 10
    MAX_STEPS = 1000
    
    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_FLOOR = (60, 45, 35)
    COLOR_FLOOR_SHADOW = (45, 30, 20)
    COLOR_WALL = (100, 110, 120)
    COLOR_WALL_TOP = (120, 130, 140)
    COLOR_EXIT = (200, 200, 80)
    
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_ACCENT = (150, 255, 150)
    COLOR_GOLD = (255, 223, 0)
    
    ENEMY_PALETTES = {
        "slime": ((80, 180, 90), (120, 220, 130)),
        "goblin": ((180, 80, 60), (220, 120, 100)),
        "archer": ((150, 100, 50), (180, 130, 80)),
        "ghost": ((180, 180, 220), (220, 220, 255)),
        "ogre": ((100, 90, 80), (130, 120, 110)),
    }

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
        
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_feedback = pygame.font.SysFont("Consolas", 16)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.player_health = 100
        self.player_max_health = 100
        self.player_attack_damage = 25
        self.room_number = 1
        self.gold_collected = 0
        self.enemies = []
        self.gold_items = []
        self.vfx = []
        self.exit_pos = (0, 0)
        self.feedback_text = ""
        self.feedback_timer = 0
        self.grid = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 100
        self.room_number = 1
        self.gold_collected = 0
        self.vfx = []
        self.feedback_text = ""
        self.feedback_timer = 0
        
        self._generate_room()
        
        return self._get_observation(), self._get_info()
    
    def _generate_room(self):
        self.enemies.clear()
        self.gold_items.clear()
        
        # 1 is walkable, 0 is wall
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = 0

        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2)
        self.exit_pos = (self.GRID_WIDTH // 2, 1)

        occupied_positions = {self.player_pos, self.exit_pos}

        # Spawn enemies
        num_enemies = min(self.room_number + self.np_random.integers(0, 2), 7)
        enemy_types = list(self.ENEMY_PALETTES.keys())
        for _ in range(num_enemies):
            pos = self._get_random_walkable_pos(occupied_positions)
            if pos:
                enemy_type = self.np_random.choice(enemy_types)
                base_health = {"slime": 20, "goblin": 30, "archer": 25, "ghost": 35, "ogre": 50}[enemy_type]
                health = base_health + 5 * (self.room_number - 1)
                
                self.enemies.append({
                    "pos": pos,
                    "type": enemy_type,
                    "health": health,
                    "max_health": health,
                    "ai_state": "idle",
                    "attack_cooldown": 0,
                })
                occupied_positions.add(pos)

        # Spawn gold
        num_gold = self.np_random.integers(1, 4)
        for _ in range(num_gold):
            pos = self._get_random_walkable_pos(occupied_positions)
            if pos:
                self.gold_items.append({"pos": pos})
                occupied_positions.add(pos)
                
    def _get_random_walkable_pos(self, occupied):
        for _ in range(100): # Avoid infinite loop
            x = self.np_random.integers(1, self.GRID_WIDTH - 1)
            y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
            if self.grid[x, y] == 1 and (x, y) not in occupied:
                return (x, y)
        return None

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0  # No survival reward per turn, only for events
        self.feedback_text = ""

        # Player action
        player_action_reward = self._handle_player_action(action)
        reward += player_action_reward

        # Enemy actions (if player didn't just win/exit)
        if not self.game_over:
            enemy_action_reward = self._update_enemies()
            reward += enemy_action_reward
        
        # Update VFX and timers
        self.vfx = [v for v in self.vfx if v.update()]
        self.feedback_timer = max(0, self.feedback_timer - 1)

        # Update game state
        self.steps += 1
        terminated = False

        if self.player_health <= 0:
            self.player_health = 0
            terminated = True
            reward = -100.0  # Death penalty
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True # Max steps reached
        
        if self.game_over and self.player_health > 0: # Reached final exit
             reward = 100.0 # Victory reward

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, action):
        movement, space_held, _ = action
        reward = 0

        # --- ATTACK ACTION ---
        if space_held == 1:
            # sound: player_attack_swing.wav
            self._add_vfx('attack', self.player_pos, life=15)
            self.feedback_text = "You attack!"
            self.feedback_timer = 30
            
            attacked_enemy = False
            for enemy in self.enemies:
                dist = math.dist(self.player_pos, enemy["pos"])
                if dist < 1.5: # Melee range
                    attacked_enemy = True
                    enemy["health"] -= self.player_attack_damage
                    self._add_vfx('hit', enemy["pos"], life=20)
                    # sound: enemy_hit.wav
                    
                    if enemy["health"] <= 0:
                        reward += 1.0  # Kill reward
                        self.gold_collected += self.np_random.integers(1, 6) # Gold from enemy
                        self.feedback_text = f"You defeated the {enemy['type']}!"
                        self.enemies.remove(enemy)
                        # sound: enemy_die.wav
            if not attacked_enemy:
                self.feedback_text = "Your attack misses!"


        # --- MOVEMENT ACTION ---
        elif movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check boundaries and walls
            if self.grid[new_pos[0], new_pos[1]] == 0:
                self.feedback_text = "You hit a wall."
                self.feedback_timer = 30
                # sound: bump_wall.wav
            # Check for enemy collision
            elif any(e["pos"] == new_pos for e in self.enemies):
                self.feedback_text = "An enemy blocks your way."
                self.feedback_timer = 30
            else:
                self.player_pos = new_pos
                # sound: player_step.wav
        
        # --- Check for post-move events ---
        # Gold pickup
        for gold in self.gold_items:
            if self.player_pos == gold["pos"]:
                reward += 0.5
                self.gold_collected += 1
                self.gold_items.remove(gold)
                self._add_vfx('pickup', self.player_pos, life=20)
                self.feedback_text = "You found 1 gold!"
                self.feedback_timer = 30
                # sound: gold_pickup.wav
                break

        # Exit
        if self.player_pos == self.exit_pos:
            if self.room_number < self.MAX_ROOMS:
                self.room_number += 1
                self._generate_room()
                self.feedback_text = f"You entered Room {self.room_number}."
                self.feedback_timer = 45
                reward += 10.0 # Reward for clearing a room
                # sound: level_up.wav
            else:
                self.game_over = True # Final victory
                self.feedback_text = "You escaped the dungeon!"
                self.feedback_timer = 120
        
        return reward

    def _update_enemies(self):
        reward = 0
        player_grid_pos = self.player_pos

        for enemy in self.enemies:
            if enemy["attack_cooldown"] > 0:
                enemy["attack_cooldown"] -= 1
                continue

            dist_to_player = math.dist(enemy["pos"], player_grid_pos)

            # AI Logic
            # Flee state
            if enemy["health"] <= enemy["max_health"] * 0.25 and enemy["type"] in ["goblin", "slime"]:
                enemy["ai_state"] = "flee"
            else:
                enemy["ai_state"] = "hunt"

            # Action based on state
            if enemy["ai_state"] == "flee":
                # Move away from player
                dx, dy = player_grid_pos[0] - enemy["pos"][0], player_grid_pos[1] - enemy["pos"][1]
                move_options = []
                if dx > 0: move_options.append((-1, 0))
                if dx < 0: move_options.append((1, 0))
                if dy > 0: move_options.append((0, -1))
                if dy < 0: move_options.append((0, 1))
                self._move_enemy(enemy, move_options)

            elif enemy["ai_state"] == "hunt":
                # Attack if in range
                attack_range = 1.5 if enemy["type"] != "archer" else 5.0
                can_attack = False
                if dist_to_player <= attack_range:
                    if enemy["type"] == "archer":
                        # Line of sight check
                        if enemy["pos"][0] == player_grid_pos[0] or enemy["pos"][1] == player_grid_pos[1]:
                            can_attack = True
                    else:
                        can_attack = True
                
                if can_attack:
                    # sound: enemy_attack.wav
                    damage = {"slime": 5, "goblin": 8, "archer": 10, "ghost": 12, "ogre": 15}[enemy["type"]]
                    self.player_health -= damage
                    reward -= 0.2 # Small penalty for getting hit
                    self.feedback_text = f"A {enemy['type']} hit you for {damage} damage!"
                    self.feedback_timer = 30
                    self._add_vfx('hit', self.player_pos, life=20, color=(255,80,80))
                    enemy["attack_cooldown"] = 1 if enemy["type"] != "ogre" else 2 # Ogres are slower
                    if enemy["type"] == "archer":
                        self._add_vfx('projectile', enemy["pos"], target_pos=player_grid_pos, life=10)

                else: # Chase
                    dx, dy = player_grid_pos[0] - enemy["pos"][0], player_grid_pos[1] - enemy["pos"][1]
                    move_options = []
                    if abs(dx) > abs(dy):
                        move_options.append((np.sign(dx), 0))
                        move_options.append((0, np.sign(dy)))
                    else:
                        move_options.append((0, np.sign(dy)))
                        move_options.append((np.sign(dx), 0))
                    self._move_enemy(enemy, move_options)
        return reward

    def _move_enemy(self, enemy, preferred_moves):
        all_moves = [(0,1), (0,-1), (1,0), (-1,0)]
        self.np_random.shuffle(all_moves)
        
        # Try preferred moves first
        for dx, dy in preferred_moves:
            if dx == 0 and dy == 0: continue
            new_pos = (enemy["pos"][0] + dx, enemy["pos"][1] + dy)
            if self._is_valid_move(new_pos, allow_player=False):
                enemy["pos"] = new_pos
                return

        # Try any random valid move if preferred fails
        for dx, dy in all_moves:
            new_pos = (enemy["pos"][0] + dx, enemy["pos"][1] + dy)
            if self._is_valid_move(new_pos, allow_player=False):
                enemy["pos"] = new_pos
                return

    def _is_valid_move(self, pos, allow_player=True):
        if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
            return False
        if self.grid[pos[0], pos[1]] == 0:
            return False
        if not allow_player and pos == self.player_pos:
            return False
        if any(e["pos"] == pos for e in self.enemies):
            return False
        return True

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = (grid_x - grid_y) * self.TILE_WIDTH_HALF + self.SCREEN_WIDTH / 2
        screen_y = (grid_x + grid_y) * self.TILE_HEIGHT_HALF + self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF)
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_pos, color, top_color, height=1.0):
        x, y = grid_pos
        h = int(height * self.TILE_HEIGHT_HALF * 2)
        sx, sy = self._iso_to_screen(x, y)
        
        wh, hh = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        # Top
        pygame.gfxdraw.filled_polygon(surface, [
            (sx, sy - h), (sx + wh, sy + hh - h),
            (sx, sy + hh * 2 - h), (sx - wh, sy + hh - h)
        ], top_color)
        
        # Left face
        pygame.draw.polygon(surface, color, [
            (sx - wh, sy + hh - h), (sx, sy + hh * 2 - h),
            (sx, sy + hh * 2), (sx - wh, sy + hh)
        ])
        
        # Right face
        pygame.draw.polygon(surface, tuple(max(0, c-20) for c in color), [
            (sx + wh, sy + hh - h), (sx, sy + hh * 2 - h),
            (sx, sy + hh * 2), (sx + wh, sy + hh)
        ])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game World ---
        # Collect all drawable objects
        draw_list = []
        for enemy in self.enemies:
            draw_list.append(('enemy', enemy))
        for gold in self.gold_items:
            draw_list.append(('gold', gold))
        draw_list.append(('player', {'pos': self.player_pos}))
        
        # Sort by grid y then x for correct isometric rendering
        draw_list.sort(key=lambda item: (item[1]['pos'][1], item[1]['pos'][0]))
        
        # Draw floor and static elements
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy + self.TILE_HEIGHT_HALF), (sx + self.TILE_WIDTH_HALF, sy + 2 * self.TILE_HEIGHT_HALF),
                    (sx, sy + 3 * self.TILE_HEIGHT_HALF), (sx - self.TILE_WIDTH_HALF, sy + 2 * self.TILE_HEIGHT_HALF)
                ]
                if self.grid[x, y] == 1:
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR)
                    if (x,y) == self.exit_pos:
                        pygame.gfxdraw.filled_circle(self.screen, sx, sy + 2*self.TILE_HEIGHT_HALF, 8, self.COLOR_EXIT)
                        pygame.gfxdraw.aacircle(self.screen, sx, sy + 2*self.TILE_HEIGHT_HALF, 8, self.COLOR_EXIT)
                else: # Wall
                    self._draw_iso_cube(self.screen, (x, y), self.COLOR_WALL, self.COLOR_WALL_TOP, height=1.5)

        # Draw dynamic objects
        bob = math.sin(self.steps * 0.2) * 2
        for item_type, item in draw_list:
            pos = item['pos']
            sx, sy = self._iso_to_screen(pos[0], pos[1])
            sy += int(self.TILE_HEIGHT_HALF * 2.5) # Center on tile

            if item_type == 'player':
                self._draw_iso_cube(self.screen, pos, self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT, height=0.6 + bob/40)
            elif item_type == 'enemy':
                palette = self.ENEMY_PALETTES[item['type']]
                height = 0.5 if item['type'] != 'ogre' else 0.8
                self._draw_iso_cube(self.screen, pos, palette[0], palette[1], height=height - bob/40)
                # Health bar for enemy
                bar_w, bar_h = 30, 4
                health_pct = item['health'] / item['max_health']
                pygame.draw.rect(self.screen, (50,0,0), (sx - bar_w/2, sy - 50, bar_w, bar_h))
                pygame.draw.rect(self.screen, (200,0,0), (sx - bar_w/2, sy - 50, bar_w * health_pct, bar_h))
            elif item_type == 'gold':
                pygame.draw.circle(self.screen, self.COLOR_GOLD, (sx, int(sy - 5 + bob)), 5)

        # Render VFX
        for vfx in self.vfx:
            vfx.draw(self.screen, self._iso_to_screen)
            
        # --- Render UI ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health Bar
        health_bar_rect = pygame.Rect(10, 10, 200, 20)
        health_pct = self.player_health / self.player_max_health
        pygame.draw.rect(self.screen, (50, 0, 0), health_bar_rect)
        pygame.draw.rect(self.screen, (200, 0, 0), (10, 10, int(200 * health_pct), 20))
        health_text = self.font_ui.render(f"HP: {self.player_health}/{self.player_max_health}", True, (255, 255, 255))
        self.screen.blit(health_text, (15, 12))
        
        # Gold and Room
        info_text = self.font_ui.render(f"Gold: {self.gold_collected} | Room: {self.room_number}/{self.MAX_ROOMS}", True, self.COLOR_GOLD)
        self.screen.blit(info_text, (10, 40))

        # Action Feedback
        if self.feedback_timer > 0:
            feedback_surf = self.font_feedback.render(self.feedback_text, True, (220, 220, 220))
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
            self.screen.blit(feedback_surf, feedback_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU DIED" if self.player_health <= 0 else "VICTORY!"
            color = (180, 0, 0) if self.player_health <= 0 else (255, 223, 0)
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _add_vfx(self, vfx_type, pos, **kwargs):
        self.vfx.append(VFX(vfx_type, pos, **kwargs))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "room": self.room_number,
            "gold": self.gold_collected,
            "health": self.player_health,
        }
        
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


class VFX:
    def __init__(self, vfx_type, grid_pos, life=20, color=None, target_pos=None):
        self.type = vfx_type
        self.grid_pos = grid_pos
        self.start_life = life
        self.life = life
        self.color = color or (255, 255, 255)
        self.target_pos = target_pos

    def update(self):
        self.life -= 1
        return self.life > 0

    def draw(self, surface, iso_converter):
        progress = self.life / self.start_life
        
        if self.type == 'hit':
            sx, sy = iso_converter(self.grid_pos[0], self.grid_pos[1])
            sy += int(GameEnv.TILE_HEIGHT_HALF * 2.5 - 20)
            radius = int(15 * (1 - progress))
            for _ in range(3):
                offset_x = random.randint(-radius, radius)
                offset_y = random.randint(-radius, radius)
                pygame.draw.circle(surface, self.color, (sx + offset_x, sy + offset_y), 1)
        elif self.type == 'attack':
            sx, sy = iso_converter(self.grid_pos[0], self.grid_pos[1])
            sy += int(GameEnv.TILE_HEIGHT_HALF * 2.5 - 10)
            radius = int(25 * (1 - progress))
            alpha = int(255 * progress)
            color = self.color + (alpha,)
            pygame.gfxdraw.aacircle(surface, sx, sy, radius, color)
        elif self.type == 'pickup':
            sx, sy = iso_converter(self.grid_pos[0], self.grid_pos[1])
            sy += int(GameEnv.TILE_HEIGHT_HALF * 2.5)
            sy -= int(20 * (1 - progress)) # Move up
            alpha = int(255 * progress)
            color = GameEnv.COLOR_GOLD
            pygame.draw.circle(surface, color + (alpha,), (sx, sy), int(8 * progress))
        elif self.type == 'projectile':
            start_sx, start_sy = iso_converter(self.grid_pos[0], self.grid_pos[1])
            start_sy += int(GameEnv.TILE_HEIGHT_HALF * 2.5 - 20)
            end_sx, end_sy = iso_converter(self.target_pos[0], self.target_pos[1])
            end_sy += int(GameEnv.TILE_HEIGHT_HALF * 2.5 - 20)
            
            interp_progress = 1 - progress
            px = int(start_sx + (end_sx - start_sx) * interp_progress)
            py = int(start_sy + (end_sy - start_sy) * interp_progress)
            
            pygame.draw.circle(surface, (255, 50, 50), (px, py), 3)