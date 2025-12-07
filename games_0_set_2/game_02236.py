import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for Monsters
class Monster:
    def __init__(self, monster_type, name, max_hp, grid_pos=None):
        self.type = monster_type
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.grid_pos = grid_pos  # [col, row]
        self.ability_cooldown = 0
        self.is_dead = False
        self.just_attacked = 0 # Countdown for attack animation
        self.just_healed = 0 # Countdown for heal animation
        self.just_damaged = 0 # Countdown for damage flash

    def take_damage(self, amount):
        self.hp = max(0, self.hp - amount)
        self.just_damaged = 10 # frames
        if self.hp == 0:
            self.is_dead = True
        return self.is_dead

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + amount)
        self.just_healed = 10 # frames

    def update(self):
        self.ability_cooldown = max(0, self.ability_cooldown - 1)
        self.just_attacked = max(0, self.just_attacked - 1)
        self.just_healed = max(0, self.just_healed - 1)
        self.just_damaged = max(0, self.just_damaged - 1)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place monster. Shift to use selected monster's ability."
    )

    game_description = (
        "Strategically arrange a team of quirky monsters to defeat a powerful boss in a side-view puzzle game."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (18, 23, 36)
    COLOR_GRID = (40, 50, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 200, 0)
    COLOR_BOSS_HP_BAR = (180, 40, 60)
    COLOR_BOSS_HP_BG = (70, 20, 30)
    COLOR_MONSTER_HP_BAR = (40, 180, 60)
    COLOR_MONSTER_HP_BG = (20, 70, 30)
    COLOR_COOLDOWN = (80, 80, 120)

    # Game Parameters
    GRID_COLS = 2
    GRID_ROWS = 3
    MAX_STEPS = 1000
    BOSS_MAX_HP = 100
    BOSS_ATTACK_DAMAGE = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_huge = pygame.font.Font(None, 72)
        self.font_damage = pygame.font.Font(None, 28)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.boss_hp = 0
        self.player_monsters = []
        self.placed_monsters = {} # grid_pos_tuple -> monster
        self.hand_monsters = []
        self.cursor_pos = [0, 0]
        self.particles = []
        self.event_logs = []
        self.boss_target = None
        self.boss_just_attacked = 0
        
        # This call to reset() will fail if the environment is not set up correctly.
        # We call it here to catch initialization errors early.
        self.reset()
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = "" # "win" or "lose"
        
        self.boss_hp = self.BOSS_MAX_HP
        self.boss_target = None
        self.boss_just_attacked = 0

        self.player_monsters = [
            Monster("knight", "Knight", 30),
            Monster("archer", "Archer", 20),
            Monster("mage", "Mage", 15),
            Monster("healer", "Healer", 20),
        ]
        self.placed_monsters = {}
        self.hand_monsters = list(self.player_monsters)
        
        self.cursor_pos = [0, 0]
        self.particles = []
        # FIX: The event log expects a tuple of (message, color).
        # The original code provided just a string, causing a ValueError on unpack.
        self.event_logs = [("Game Start! Place your first monster.", self.COLOR_TEXT)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = False
        action_taken = False
        
        # --- 1. Player Action ---
        self._handle_cursor_movement(movement)
        
        if space_held:
            action_taken = self._action_place_monster()
            if not action_taken:
                reward = -0.2 # Invalid placement
                self.log_event("Invalid placement!", (255, 100, 100))
                action_taken = True # Consumes turn
        elif shift_held:
            reward, action_taken = self._action_use_ability()
            if not action_taken:
                reward = -0.2 # Invalid ability use
                self.log_event("Cannot use ability!", (255, 100, 100))
                action_taken = True # Consumes turn
        elif movement == 0: # No-op
            reward = -0.2
            self.log_event("Player waits...", (200, 200, 200))
            action_taken = True

        # --- 2. Turn Resolution (if an action was taken) ---
        if action_taken:
            # Update monster cooldowns and effects at start of resolution
            for m in self.placed_monsters.values():
                m.update()

            # A. Resolve player passive attacks
            damage_from_passives = self._resolve_passive_attacks()
            reward += damage_from_passives
            
            # B. Check for player win
            if self.boss_hp <= 0:
                reward += 100
                self.score += reward
                self.game_over = True
                self.win_state = "win"
                terminated = True
                return self._get_observation(), reward, terminated, False, self._get_info()
            
            # C. Boss turn
            self._resolve_boss_attack()

            # D. Check for player loss
            if all(m.is_dead for m in self.player_monsters if m not in self.hand_monsters):
                reward -= 100
                self.score += reward
                self.game_over = True
                self.win_state = "lose"
                terminated = True
                return self._get_observation(), reward, terminated, False, self._get_info()

        # --- 3. Check for step limit ---
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if self.win_state == "": # If not already won/lost
                reward -= 50 # Penalty for timeout
                self.game_over = True
                self.win_state = "lose"

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False in this environment
            self._get_info()
        )

    def _handle_cursor_movement(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        if movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1) # Down
        if movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        if movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1) # Right

    def _action_place_monster(self):
        pos_tuple = tuple(self.cursor_pos)
        if pos_tuple in self.placed_monsters:
            return False # Slot occupied
        if not self.hand_monsters:
            return False # Hand empty

        monster_to_place = self.hand_monsters.pop(0)
        monster_to_place.grid_pos = list(self.cursor_pos)
        self.placed_monsters[pos_tuple] = monster_to_place
        self.log_event(f"Placed {monster_to_place.name}!", (150, 255, 150))
        return True

    def _action_use_ability(self):
        pos_tuple = tuple(self.cursor_pos)
        monster = self.placed_monsters.get(pos_tuple)
        if not monster or monster.ability_cooldown > 0 or monster.is_dead:
            return 0, False

        reward = 5.0
        
        # Knight: Taunt (Passive ability, no 'shift' action)
        # For this design, Knight has no active ability.

        # Archer: Piercing Shot
        if monster.type == "archer":
            damage = 10
            self.boss_hp = max(0, self.boss_hp - damage)
            monster.ability_cooldown = 4 # 3 turns + current
            monster.just_attacked = 10
            self._create_particles(10, self._get_monster_center(monster.grid_pos), self._get_boss_center(), (50, 255, 50), 2, 30, 'arrow')
            self._create_damage_number(damage, self._get_boss_center())
            self.log_event(f"Archer used Piercing Shot for {damage} dmg!", (50, 255, 50))
            reward += damage
        # Mage: Fireball
        elif monster.type == "mage":
            damage = 18
            self.boss_hp = max(0, self.boss_hp - damage)
            monster.ability_cooldown = 6 # 5 turns + current
            monster.just_attacked = 10
            self._create_particles(1, self._get_monster_center(monster.grid_pos), self._get_boss_center(), (255, 100, 0), 15, 30, 'ball')
            self._create_damage_number(damage, self._get_boss_center())
            self.log_event(f"Mage used Fireball for {damage} dmg!", (255, 100, 0))
            reward += damage
        # Healer: Heal
        elif monster.type == "healer":
            target_monster = self.placed_monsters.get(pos_tuple)
            if target_monster and not target_monster.is_dead:
                heal_amount = 10
                target_monster.heal(heal_amount)
                monster.ability_cooldown = 3 # 2 turns + current
                monster.just_attacked = 10 # 'using ability' animation
                self._create_particles(20, self._get_monster_center(monster.grid_pos), self._get_monster_center(target_monster.grid_pos), (100, 255, 100), 2, 15, 'heal')
                self._create_damage_number(heal_amount, self._get_monster_center(target_monster.grid_pos), (100, 255, 100))
                self.log_event(f"Healer restored {heal_amount} HP to {target_monster.name}!", (100, 255, 100))
            else: # Should not happen if cursor is on healer
                return -0.2, False
        else: # Monster has no ability
            return 0, False
            
        return reward, True

    def _resolve_passive_attacks(self):
        total_damage = 0
        for m in self.placed_monsters.values():
            if m.is_dead: continue
            
            # Knight: Attacks if in front column
            if m.type == "knight" and m.grid_pos[0] == 1:
                damage = 5
                self.boss_hp = max(0, self.boss_hp - damage)
                total_damage += damage
                m.just_attacked = 10
                # Short, sharp particle effect
                self._create_particles(5, self._get_monster_center(m.grid_pos), self._get_boss_center(), (200, 200, 255), 1, 10, 'line')
                self._create_damage_number(damage, self._get_boss_center())
                self.log_event(f"Knight dealt {damage} passive damage.", (200, 200, 255))
        return total_damage

    def _resolve_boss_attack(self):
        self.boss_just_attacked = 10
        living_monsters = [m for m in self.placed_monsters.values() if not m.is_dead]
        if not living_monsters:
            return

        target = self.np_random.choice(living_monsters)
        self.boss_target = target
        is_fatal = target.take_damage(self.BOSS_ATTACK_DAMAGE)
        
        self._create_particles(15, self._get_boss_center(), self._get_monster_center(target.grid_pos), (255, 50, 50), 3, 20, 'claw')
        self._create_damage_number(self.BOSS_ATTACK_DAMAGE, self._get_monster_center(target.grid_pos))
        
        log_msg = f"Boss attacks {target.name} for {self.BOSS_ATTACK_DAMAGE} dmg."
        if is_fatal:
            log_msg += f" {target.name} has been defeated!"
            # We don't remove from placed_monsters here, just mark as dead.
            # This allows effects to still see the dead monster if needed for one frame.
            # A cleanup phase could remove them later if memory becomes an issue.
        self.log_event(log_msg, (255, 100, 100))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_details()
        self._render_grid()
        self._render_boss()
        self._render_monsters()
        self._update_and_render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "boss_hp": self.boss_hp,
            "monsters_alive": sum(1 for m in self.player_monsters if not m.is_dead)
        }

    # --- Rendering Methods ---
    def _render_background_details(self):
        # Darker pillars for depth
        for i in range(5):
            x = 100 + i * 120
            pygame.draw.rect(self.screen, (25, 33, 48), (x, 50, 20, 300))
            pygame.draw.rect(self.screen, (30, 40, 58), (x-5, 45, 30, 10))
            pygame.draw.rect(self.screen, (30, 40, 58), (x-5, 350, 30, 10))

    def _render_grid(self):
        self.grid_base_x, self.grid_base_y = 50, 100
        self.cell_w, self.cell_h = 80, 80
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = (self.grid_base_x + c * self.cell_w, self.grid_base_y + r * self.cell_h, self.cell_w, self.cell_h)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        c, r = self.cursor_pos
        cursor_rect = (self.grid_base_x + c * self.cell_w, self.grid_base_y + r * self.cell_h, self.cell_w, self.cell_h)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_boss(self):
        boss_center = self._get_boss_center()
        radius = 60 + math.sin(self.steps * 0.1) * 5 # Throbbing effect
        eye_radius = 15
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, boss_center[0], boss_center[1], int(radius), (50, 50, 60))
        pygame.gfxdraw.aacircle(self.screen, boss_center[0], boss_center[1], int(radius), (80, 80, 90))
        
        # Eye
        eye_pos = list(boss_center)
        if self.boss_target and not self.boss_target.is_dead:
            target_pos = self._get_monster_center(self.boss_target.grid_pos)
            angle = math.atan2(target_pos[1] - eye_pos[1], target_pos[0] - eye_pos[0])
            eye_pos[0] += math.cos(angle) * 20
            eye_pos[1] += math.sin(angle) * 20

        pygame.gfxdraw.filled_circle(self.screen, int(eye_pos[0]), int(eye_pos[1]), eye_radius, (255, 0, 0))
        pygame.gfxdraw.aacircle(self.screen, int(eye_pos[0]), int(eye_pos[1]), eye_radius, (255, 100, 100))
        pupil_radius = 5 + math.sin(self.steps * 0.2) * 2
        pygame.gfxdraw.filled_circle(self.screen, int(eye_pos[0]), int(eye_pos[1]), int(pupil_radius), (0, 0, 0))

    def _render_monsters(self):
        for pos_tuple, monster in self.placed_monsters.items():
            if monster.is_dead: continue
            
            center = self._get_monster_center(pos_tuple)
            size = 20
            
            # Flash if damaged
            flash_color = (255, 50, 50, int(monster.just_damaged * 25.5))
            if monster.just_damaged > 0:
                s = pygame.Surface((size*2.5, size*2.5), pygame.SRCALPHA)
                pygame.draw.circle(s, flash_color, (size*1.25, size*1.25), size*1.2)
                self.screen.blit(s, (center[0] - size*1.25, center[1] - size*1.25))

            # Draw monster shape
            if monster.type == "knight":
                pygame.draw.rect(self.screen, (200, 200, 210), (center[0]-size//2, center[1]-size//2, size, size))
            elif monster.type == "archer":
                pygame.draw.polygon(self.screen, (50, 200, 50), [(center[0], center[1]-size//2), (center[0]-size//2, center[1]+size//2), (center[0]+size//2, center[1]+size//2)])
            elif monster.type == "mage":
                y_offset = math.sin(self.steps * 0.15 + pos_tuple[0]) * 3
                pygame.draw.circle(self.screen, (180, 50, 220), (center[0], int(center[1] + y_offset)), size // 2)
            elif monster.type == "healer":
                pygame.draw.rect(self.screen, (100, 150, 255), (center[0]-size//2, center[1]-size//4, size, size//2))
                pygame.draw.rect(self.screen, (100, 150, 255), (center[0]-size//4, center[1]-size//2, size//2, size))

            # Health bar
            hp_bar_y = center[1] + size//2 + 5
            hp_ratio = monster.hp / monster.max_hp
            pygame.draw.rect(self.screen, self.COLOR_MONSTER_HP_BG, (center[0]-20, hp_bar_y, 40, 5))
            pygame.draw.rect(self.screen, self.COLOR_MONSTER_HP_BAR, (center[0]-20, hp_bar_y, int(40 * hp_ratio), 5))

            # Cooldown indicator
            if monster.ability_cooldown > 1:
                cd_text = self.font_main.render(str(monster.ability_cooldown - 1), True, self.COLOR_TEXT)
                cd_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
                cd_surf.fill((*self.COLOR_COOLDOWN, 180))
                cd_surf.blit(cd_text, (cd_surf.get_width() // 2 - cd_text.get_width() // 2, cd_surf.get_height() // 2 - cd_text.get_height() // 2))
                self.screen.blit(cd_surf, (center[0] - 10, center[1] - 10))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Boss HP Bar
        hp_text = self.font_main.render(f"BOSS HP: {int(self.boss_hp)} / {self.BOSS_MAX_HP}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (self.screen.get_width() - hp_text.get_width() - 10, 10))
        hp_ratio = self.boss_hp / self.BOSS_MAX_HP
        pygame.draw.rect(self.screen, self.COLOR_BOSS_HP_BG, (340, 40, 290, 20))
        pygame.draw.rect(self.screen, self.COLOR_BOSS_HP_BAR, (340, 40, int(290 * hp_ratio), 20))

        # Hand UI
        hand_y = 360
        if self.hand_monsters:
            next_monster = self.hand_monsters[0]
            hand_text = self.font_main.render(f"Next up: {next_monster.name}", True, self.COLOR_TEXT)
            self.screen.blit(hand_text, (10, hand_y))
        else:
            hand_text = self.font_main.render("Hand empty", True, self.COLOR_TEXT)
            self.screen.blit(hand_text, (10, hand_y))
        
        # Event Log
        for i, (msg, color) in enumerate(self.event_logs[-3:]):
            log_text = self.font_main.render(msg, True, color)
            self.screen.blit(log_text, (250, 345 + i * 18))

    def _render_game_over(self):
        overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win_state == "win" else "YOU LOSE"
        color = (100, 255, 100) if self.win_state == "win" else (255, 100, 100)
        
        text = self.font_huge.render(message, True, color)
        text_rect = text.get_rect(center=(320, 180))
        overlay.blit(text, text_rect)
        
        score_text = self.font_large.render(f"Final Score: {int(self.score)}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(center=(320, 240))
        overlay.blit(score_text, score_rect)
        
        self.screen.blit(overlay, (0, 0))

    # --- Particles & Effects ---
    def _create_particles(self, count, start_pos, end_pos, color, size, lifetime, p_type='line'):
        for _ in range(count):
            angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            angle += self.np_random.uniform(-0.5, 0.5)
            speed = self.np_random.uniform(3, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(start_pos), 'vel': vel, 'color': color,
                'size': size, 'max_life': lifetime, 'life': lifetime, 'type': p_type
            })

    def _create_damage_number(self, amount, pos, color=(255, 80, 80)):
        self.particles.append({
            'pos': [pos[0], pos[1] - 20], 'vel': [self.np_random.uniform(-0.5, 0.5), -1],
            'text': str(int(amount)), 'color': color, 'max_life': 40, 'life': 40, 'type': 'text'
        })

    def _update_and_render_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue

            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))

            if p['type'] == 'text':
                text_surf = self.font_damage.render(p['text'], True, p['color'])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, pos_int)
            elif p['type'] == 'ball':
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color_with_alpha, (p['size'], p['size']), p['size'])
                self.screen.blit(s, (pos_int[0]-p['size'], pos_int[1]-p['size']))
            elif p['type'] == 'heal':
                pygame.draw.line(self.screen, color_with_alpha, (pos_int[0]-p['size'], pos_int[1]), (pos_int[0]+p['size'], pos_int[1]), 2)
                pygame.draw.line(self.screen, color_with_alpha, (pos_int[0], pos_int[1]-p['size']), (pos_int[0], pos_int[1]+p['size']), 2)
            else: # line, arrow, claw
                end_pos = (p['pos'][0] + p['vel'][0]*2, p['pos'][1] + p['vel'][1]*2)
                pygame.draw.aaline(self.screen, p['color'], p['pos'], end_pos, int(p['size'] * (p['life']/p['max_life'])))

    # --- Helpers ---
    def _get_monster_center(self, grid_pos):
        c, r = grid_pos
        return (
            self.grid_base_x + c * self.cell_w + self.cell_w // 2,
            self.grid_base_y + r * self.cell_h + self.cell_h // 2,
        )

    def _get_boss_center(self):
        return (490, 200)

    def log_event(self, message, color=(255, 255, 255)):
        self.event_logs.append((message, color))
        if len(self.event_logs) > 50:
            self.event_logs.pop(0)

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        import pygame
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Monster Boss Puzzle")
        
        terminated = False
        clock = pygame.time.Clock()
        
        # Game loop
        while not terminated:
            movement, space, shift = 0, 0, 0
            
            action_taken_this_frame = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    action_taken_this_frame = True
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space = 1
                    elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                    else:
                        action_taken_this_frame = False # No relevant key pressed

            if action_taken_this_frame:
                action = [movement, space, shift]
                obs, reward, term, trunc, info = env.step(action)
                terminated = term
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) 
            
        pygame.quit()
    except pygame.error as e:
        print(f"Pygame display could not be initialized: {e}")
        print("This is expected in a headless environment. The GameEnv class is still valid.")
        # You can add a simple test here to verify the env works headlessly
        env = GameEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Headless environment step successful.")