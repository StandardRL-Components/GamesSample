
# Generated: 2025-08-28T03:04:10.861699
# Source Brief: brief_04810.md
# Brief Index: 4810

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. In combat, press Space to Attack or Shift to Defend. Moving away from an enemy in combat will provoke an attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based roguelike RPG. Navigate the dungeon, defeat enemies to gain experience, and find the exit to proceed to the next level. Reach the exit on level 5 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.TILE_SIZE = 20
        self.MAX_STEPS = 1000
        self.MAX_LEVEL = 5

        # Colors
        self.COLOR_BG = (10, 5, 15)
        self.COLOR_WALL = (40, 30, 50)
        self.COLOR_FLOOR = (70, 60, 80)
        self.COLOR_HERO = (50, 255, 50)
        self.COLOR_HERO_GLOW = (50, 255, 50, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_EXIT = (100, 100, 255)
        self.COLOR_EXIT_GLOW = (100, 100, 255, 80)
        self.COLOR_UI_BG = (30, 30, 40, 200)
        self.COLOR_HP_BAR = (200, 0, 0)
        self.COLOR_XP_BAR = (200, 200, 0)
        self.COLOR_BAR_BG = (50, 50, 50)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_YELLOW = (255, 255, 0)

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
        try:
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 14)
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_combat = pygame.font.Font(pygame.font.get_default_font(), 16)
        except:
            # Fallback if default font is not found (e.g. in minimal container)
            self.font_small = pygame.font.SysFont("sans", 14)
            self.font_large = pygame.font.SysFont("sans", 18)
            self.font_combat = pygame.font.SysFont("sans", 16)
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        self.hero_max_hp = 20
        self.hero_hp = self.hero_max_hp
        self.hero_xp = 0
        self.hero_xp_needed = 10
        self.hero_level = 1
        self.hero_attack = 2
        self.hero_is_defending = False
        
        self.in_combat_with = None
        
        self.floating_texts = []
        self.particles = []

        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1

        self.hero_is_defending = False # Reset defense state each turn

        # --- COMBAT LOGIC ---
        if self.in_combat_with:
            enemy = self.in_combat_with
            player_action_taken = False
            
            # Player tries to move away from combat
            if movement != 0:
                # Enemy gets an attack of opportunity
                self._add_floating_text("Flee!", self.hero_pos, self.COLOR_YELLOW)
                self._enemy_attack(enemy)
                self.in_combat_with = None
                player_action_taken = True
                if not self.game_over:
                    self._handle_movement(movement) # Complete the move
            
            # Player attacks
            elif space_pressed:
                # sfx: player_attack_sfx()
                damage = self.hero_attack
                enemy["hp"] -= damage
                self._add_floating_text(str(damage), enemy["pos"], self.COLOR_WHITE)
                self._create_particles(enemy["pos"], 10, self.COLOR_ENEMY)
                
                if enemy["hp"] <= 0:
                    # sfx: enemy_death_sfx()
                    reward += 1.0 # Defeated enemy reward
                    self.score += 10
                    xp_gain = int(10 * (1 + (self.level - 1) * 0.1))
                    self.hero_xp += xp_gain
                    self._add_floating_text(f"+{xp_gain} XP", self.hero_pos, self.COLOR_YELLOW)
                    self.enemies.remove(enemy)
                    self.in_combat_with = None
                    self._check_level_up()
                player_action_taken = True

            # Player defends
            elif shift_pressed:
                # sfx: player_defend_sfx()
                self.hero_is_defending = True
                self._add_floating_text("Defend", self.hero_pos, (150, 150, 255))
                player_action_taken = True

            # Enemy turn (if combat is still ongoing)
            if self.in_combat_with and player_action_taken:
                self._enemy_attack(enemy)

        # --- MOVEMENT LOGIC ---
        elif movement != 0:
            self._handle_movement(movement)

        # --- ENEMY MOVEMENT (if not in combat) ---
        if not self.in_combat_with:
            for enemy in self.enemies:
                self._move_enemy(enemy)
        
        # --- UPDATE EFFECTS ---
        self._update_particles()
        self._update_floating_texts()

        # --- CHECK TERMINATION ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over:
            reward += -100 # Death penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        dx, dy = 0, 0
        if movement_action == 1: dy = -1  # Up
        elif movement_action == 2: dy = 1   # Down
        elif movement_action == 3: dx = -1  # Left
        elif movement_action == 4: dx = 1   # Right

        target_x, target_y = self.hero_pos[0] + dx, self.hero_pos[1] + dy
        
        if 0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT and self.grid[target_y][target_x] == 1:
            self.hero_pos = (target_x, target_y)
            self._create_particles(self.hero_pos, 2, self.COLOR_FLOOR, speed=0.5)

            # Check for enemy collision
            for enemy in self.enemies:
                if enemy["pos"] == self.hero_pos:
                    self.in_combat_with = enemy
                    self._add_floating_text("!", enemy["pos"], self.COLOR_ENEMY)
                    # sfx: combat_start_sfx()
                    break
            
            # Check for exit collision
            if self.hero_pos == self.exit_pos:
                if self.level == self.MAX_LEVEL:
                    self.game_over = True
                    self.score += 1000
                    # reward is handled outside as a special case
                    # sfx: victory_sfx()
                else:
                    self.level += 1
                    self.score += 100
                    # sfx: level_up_sfx()
                    self._generate_level()

    def _enemy_attack(self, enemy):
        # sfx: enemy_attack_sfx()
        damage = enemy["damage"]
        if self.hero_is_defending:
            damage = math.ceil(damage / 2)
        
        self.hero_hp -= damage
        self._add_floating_text(str(damage), self.hero_pos, self.COLOR_ENEMY)
        self._create_particles(self.hero_pos, 10, self.COLOR_HERO)

        if self.hero_hp <= 0:
            self.hero_hp = 0
            self.game_over = True
            # sfx: player_death_sfx()
            
    def _check_level_up(self):
        while self.hero_xp >= self.hero_xp_needed:
            # sfx: player_level_up_sfx()
            self.hero_level += 1
            self.hero_xp -= self.hero_xp_needed
            self.hero_xp_needed = int(self.hero_xp_needed * 1.5)
            self.hero_max_hp += 5
            self.hero_hp = self.hero_max_hp # Full heal on level up
            self.hero_attack += 1
            self._add_floating_text("LEVEL UP!", self.hero_pos, self.COLOR_YELLOW, size='large')
            self._create_particles(self.hero_pos, 30, self.COLOR_YELLOW, speed=2)

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Carve a main path
        px, py = 1, self.np_random.integers(1, self.GRID_HEIGHT - 1)
        self.hero_pos = (px, py)
        self.grid[py][px] = 1
        path = [(px,py)]
        
        while px < self.GRID_WIDTH - 2:
            # Move mostly right
            move = self.np_random.choice(['r', 'r', 'r', 'u', 'd'])
            if move == 'r' and px < self.GRID_WIDTH - 2:
                px += 1
            elif move == 'u' and py > 1:
                py -= 1
            elif move == 'd' and py < self.GRID_HEIGHT - 2:
                py += 1
            self.grid[py][px] = 1
            if (px,py) not in path: path.append((px,py))

        self.exit_pos = (px, py)

        # Carve side paths
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT // 20):
            sx, sy = random.choice(path)
            for _ in range(self.np_random.integers(5, 15)):
                nx, ny = sx + self.np_random.integers(-1, 2), sy + self.np_random.integers(-1, 2)
                if 0 < nx < self.GRID_WIDTH -1 and 0 < ny < self.GRID_HEIGHT - 1:
                    self.grid[ny][nx] = 1
                    sx, sy = nx, ny
                    if (sx,sy) not in path: path.append((sx,sy))

        # Place enemies
        self.enemies = []
        num_enemies = self.np_random.integers(self.level + 2, self.level + 5)
        for _ in range(num_enemies):
            while True:
                ex, ey = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
                # Ensure not on hero start, exit, or wall
                if self.grid[ey][ex] == 1 and (ex, ey) != self.hero_pos and (ex, ey) != self.exit_pos:
                    is_occupied = any(e['pos'] == (ex, ey) for e in self.enemies)
                    if not is_occupied:
                        base_hp = 10
                        base_damage = 1
                        enemy_hp = int(base_hp + (self.level - 1))
                        enemy_damage = int(base_damage + (self.level - 1))
                        self.enemies.append({
                            "pos": (ex, ey), 
                            "hp": enemy_hp, 
                            "max_hp": enemy_hp, 
                            "damage": enemy_damage,
                            "patrol_origin": (ex, ey),
                            "patrol_dir": self.np_random.choice(['x', 'y']),
                            "patrol_step": 0
                        })
                        break
        self.in_combat_with = None

    def _move_enemy(self, enemy):
        ox, oy = enemy["patrol_origin"]
        ex, ey = enemy["pos"]
        
        # Simple patrol logic: move back and forth 2 tiles from origin
        if enemy["patrol_dir"] == 'x':
            if enemy["patrol_step"] < 2: target_pos = (ex + 1, ey)
            else: target_pos = (ex - 1, ey)
        else: # 'y'
            if enemy["patrol_step"] < 2: target_pos = (ex, ey + 1)
            else: target_pos = (ex, ey - 1)
        
        # Check if target is valid and not hero pos
        tx, ty = target_pos
        if (0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT and
            self.grid[ty][tx] == 1 and target_pos != self.hero_pos):
            enemy["pos"] = target_pos
        
        enemy["patrol_step"] = (enemy["patrol_step"] + 1) % 4


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_dungeon()
        self._render_entities()
        self._render_particles()
        self._render_floating_texts()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_dungeon(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_FLOOR if self.grid[y][x] == 1 else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, rect)

    def _render_entities(self):
        # Exit portal
        ex, ey = self.exit_pos
        center_x, center_y = int((ex + 0.5) * self.TILE_SIZE), int((ey + 0.5) * self.TILE_SIZE)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        
        radius_glow = int(self.TILE_SIZE * (0.6 + pulse * 0.2))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius_glow, self.COLOR_EXIT_GLOW)
        
        radius_main = int(self.TILE_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius_main, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius_main, self.COLOR_WHITE)

        # Enemies
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            center_x, center_y = int((ex + 0.5) * self.TILE_SIZE), int((ey + 0.5) * self.TILE_SIZE)
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.TILE_SIZE * 0.6), self.COLOR_ENEMY_GLOW)
            
            # Body
            size = int(self.TILE_SIZE * 0.7)
            rect = pygame.Rect(0, 0, size, size)
            rect.center = (center_x, center_y)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)
            
            # Health bar
            if enemy["hp"] < enemy["max_hp"]:
                bar_w = self.TILE_SIZE
                bar_h = 4
                bar_x = ex * self.TILE_SIZE
                bar_y = ey * self.TILE_SIZE - bar_h - 2
                self._render_bar(bar_x, bar_y, bar_w, bar_h, enemy["hp"] / enemy["max_hp"], self.COLOR_HP_BAR, self.COLOR_BAR_BG)

        # Hero
        hx, hy = self.hero_pos
        center_x, center_y = int((hx + 0.5) * self.TILE_SIZE), int((hy + 0.5) * self.TILE_SIZE)
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.TILE_SIZE * 0.8), self.COLOR_HERO_GLOW)
        
        # Body
        size = int(self.TILE_SIZE * 0.8)
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (center_x, center_y)
        pygame.draw.rect(self.screen, self.COLOR_HERO, rect, border_radius=3)

    def _render_ui(self):
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Health Bar
        hp_text = self.font_small.render("HP", True, self.COLOR_WHITE)
        self.screen.blit(hp_text, (10, 12))
        self._render_bar(40, 10, 150, 20, self.hero_hp / self.hero_max_hp, self.COLOR_HP_BAR, self.COLOR_BAR_BG)
        hp_val_text = self.font_small.render(f"{self.hero_hp}/{self.hero_max_hp}", True, self.COLOR_WHITE)
        self.screen.blit(hp_val_text, (95, 12))

        # XP Bar
        xp_text = self.font_small.render("XP", True, self.COLOR_WHITE)
        self.screen.blit(xp_text, (200, 12))
        xp_ratio = self.hero_xp / self.hero_xp_needed if self.hero_xp_needed > 0 else 1
        self._render_bar(230, 10, 150, 20, xp_ratio, self.COLOR_XP_BAR, self.COLOR_BAR_BG)
        xp_val_text = self.font_small.render(f"LVL {self.hero_level}", True, self.COLOR_WHITE)
        self.screen.blit(xp_val_text, (275, 12))

        # Level and Score
        level_text = self.font_large.render(f"Dungeon Level: {self.level}", True, self.COLOR_WHITE)
        self.screen.blit(level_text, (400, 10))
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

    def _render_bar(self, x, y, w, h, progress, fg_color, bg_color):
        progress = max(0, min(1, progress))
        pygame.draw.rect(self.screen, bg_color, (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, fg_color, (x, y, int(w * progress), h), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (x, y, w, h), 1, border_radius=4)

    def _add_floating_text(self, text, grid_pos, color, duration=30, size='normal'):
        font = self.font_large if size == 'large' else self.font_combat
        self.floating_texts.append({
            "text": text,
            "pos": [ (grid_pos[0] + 0.5) * self.TILE_SIZE, (grid_pos[1] + 0.5) * self.TILE_SIZE ],
            "color": color,
            "timer": duration,
            "max_timer": duration,
            "font": font
        })

    def _update_floating_texts(self):
        for ft in self.floating_texts[:]:
            ft["timer"] -= 1
            ft["pos"][1] -= 0.5 # Move up
            if ft["timer"] <= 0:
                self.floating_texts.remove(ft)

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            alpha = int(255 * (ft["timer"] / ft["max_timer"]))
            text_surf = ft["font"].render(ft["text"], True, ft["color"])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(ft["pos"][0]), int(ft["pos"][1])))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_pos, count, color, speed=1.5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({
                "pos": [ (grid_pos[0] + 0.5) * self.TILE_SIZE, (grid_pos[1] + 0.5) * self.TILE_SIZE ],
                "vel": vel,
                "color": color,
                "timer": self.np_random.integers(15, 25)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["timer"] -= 1
            if p["timer"] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(3 * (p["timer"] / 25)))
            pygame.draw.rect(self.screen, p["color"], (*p["pos"], size, size))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "hero_hp": self.hero_hp,
            "hero_xp": self.hero_xp,
            "hero_level": self.hero_level,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:")
    print(f"  Info: {info}")
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: Action={action}")
        print(f"  Reward={reward}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
            
    env.close()