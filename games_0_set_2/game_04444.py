
# Generated: 2025-08-28T02:25:36.413708
# Source Brief: brief_04444.md
# Brief Index: 4444

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in your last moved direction. Survive 3 stages."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a grid-based arena. Collect potions and survive to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    GRID_AREA_SIZE = 360
    CELL_SIZE = GRID_AREA_SIZE // GRID_SIZE
    GRID_TOP_LEFT = ((SCREEN_WIDTH - GRID_AREA_SIZE) // 2, (SCREEN_HEIGHT - GRID_AREA_SIZE) // 2)

    MAX_STAGES = 3
    TURN_LIMIT_PER_STAGE = 100

    PLAYER_MAX_HEALTH = 10
    MONSTER_BASE_HEALTH = 2
    POTION_HEAL_AMOUNT = 3
    MONSTER_DAMAGE = 2

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (45, 45, 60)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_ACCENT = (150, 255, 200)
    COLOR_MONSTER = (255, 80, 80)
    COLOR_MONSTER_ACCENT = (255, 150, 150)
    COLOR_POTION = (80, 150, 255)
    COLOR_POTION_ACCENT = (150, 200, 255)
    COLOR_ATTACK = (255, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_HIGH = (0, 200, 0)
    COLOR_HEALTH_MED = (255, 255, 0)
    COLOR_HEALTH_LOW = (255, 0, 0)

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)

        self.player_pos = (0, 0)
        self.player_health = 0
        self.player_facing = (0, 1)  # (dx, dy) Down by default
        self.monsters = []
        self.potions = []
        self.particles = []
        self.damage_indicators = []
        
        self.stage = 1
        self.score = 0
        self.turns_taken = 0
        
        self.inter_stage_timer = 0
        self.game_over_timer = 0
        self.game_won = False
        self.terminated = False

        self.screen_shake = 0

        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.stage = 1
        
        self.inter_stage_timer = 0
        self.game_over_timer = 0
        self.game_won = False
        self.terminated = False

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.player_health = self.PLAYER_MAX_HEALTH
        self.turns_taken = 0
        
        occupied_cells = set()
        
        # Place player
        self.player_pos = (self.GRID_SIZE // 2, self.GRID_SIZE - 2)
        self.player_facing = (0, -1) # Start facing up
        occupied_cells.add(self.player_pos)

        # Spawn monsters
        self.monsters = []
        monster_health = self.MONSTER_BASE_HEALTH + (self.stage - 1)
        for i in range(5):
            pos = self._get_random_empty_cell(occupied_cells)
            self.monsters.append({
                "pos": pos,
                "health": monster_health,
                "max_health": monster_health,
                "id": self.np_random.integers(10000),
                "hit_timer": 0,
            })
            occupied_cells.add(pos)
            
        # Spawn potions
        self.potions = []
        for _ in range(2):
            pos = self._get_random_empty_cell(occupied_cells)
            self.potions.append(pos)
            occupied_cells.add(pos)

        self.particles = []
        self.damage_indicators = []

    def _get_random_empty_cell(self, occupied_cells):
        while True:
            pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if pos not in occupied_cells:
                return pos

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, self.terminated, False, self._get_info()
            
        reward = 0
        
        # Handle message screens (stage clear, game over)
        if self.inter_stage_timer > 0:
            self.inter_stage_timer -= 1
            if self.inter_stage_timer == 0:
                self.stage += 1
                self._setup_stage()
            return self._get_observation(), 0, self.terminated, False, self._get_info()
        
        if self.game_over_timer > 0:
            self.game_over_timer -= 1
            if self.game_over_timer == 0:
                self.terminated = True
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        self.turns_taken += 1
        movement_action = action[0]
        attack_action = action[1] == 1

        # --- Player Turn ---
        reward += self._handle_player_turn(movement_action, attack_action)
        
        # --- Monster Turn ---
        self._handle_monster_turn()

        # --- Update Game State & Check Conditions ---
        state_reward, terminated = self._update_game_state()
        reward += state_reward
        self.terminated = terminated

        # --- Update Visual Effects ---
        for m in self.monsters:
            if m["hit_timer"] > 0:
                m["hit_timer"] -= 1

        return self._get_observation(), reward, self.terminated, False, self._get_info()

    def _handle_player_turn(self, movement_action, attack_action):
        reward = 0
        
        # 1. Handle Movement
        old_pos = self.player_pos
        dx, dy = 0, 0
        if movement_action == 1: dy = -1  # Up
        elif movement_action == 2: dy = 1   # Down
        elif movement_action == 3: dx = -1  # Left
        elif movement_action == 4: dx = 1   # Right
        
        if (dx, dy) != (0, 0):
            self.player_facing = (dx, dy)
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Check boundaries
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                # Check monster collision
                if not any(m['pos'] == new_pos for m in self.monsters):
                    self.player_pos = new_pos
        
        # Movement-based reward
        if movement_action == 0 and not attack_action:
            reward -= 0.1 # Penalty for doing nothing
        
        # 2. Handle Attack
        if attack_action:
            attack_pos = (self.player_pos[0] + self.player_facing[0], self.player_pos[1] + self.player_facing[1])
            self._create_particles(self.player_pos, self.player_facing, self.COLOR_ATTACK, 10, 1) # sfx: player_attack

            for monster in self.monsters:
                if monster['pos'] == attack_pos:
                    monster['health'] -= 1
                    monster['hit_timer'] = 5 # frames to flash
                    reward += 1 # Reward for hitting
                    self._create_damage_indicator(monster['pos'], 1)
                    # sfx: monster_hit
                    if monster['health'] <= 0:
                        reward += 2 # Reward for defeating
                        self.score += 100
                        self._create_particles(monster['pos'], (0,0), self.COLOR_MONSTER, 30, 2) # sfx: monster_death
            
            self.monsters = [m for m in self.monsters if m['health'] > 0]

        # 3. Handle Potion Collection
        if self.player_pos in self.potions:
            self.potions.remove(self.player_pos)
            self.player_health = min(self.PLAYER_MAX_HEALTH, self.player_health + self.POTION_HEAL_AMOUNT)
            reward += 1
            self.score += 50
            self._create_damage_indicator(self.player_pos, f"+{self.POTION_HEAL_AMOUNT}", self.COLOR_POTION) # sfx: potion_collect

        return reward

    def _handle_monster_turn(self):
        monster_moves = []
        for monster in self.monsters:
            dist_x = self.player_pos[0] - monster['pos'][0]
            dist_y = self.player_pos[1] - monster['pos'][1]
            
            # Move if player is within 3 cells (Manhattan distance)
            if abs(dist_x) + abs(dist_y) <= 3 + (self.stage - 1) * 0.5:
                # Simple greedy movement
                if abs(dist_x) > abs(dist_y):
                    new_pos = (monster['pos'][0] + np.sign(dist_x), monster['pos'][1])
                elif abs(dist_y) > 0:
                    new_pos = (monster['pos'][0], monster['pos'][1] + np.sign(dist_y))
                else:
                    new_pos = monster['pos']
                
                monster_moves.append((monster, new_pos))

        # Resolve move conflicts and update positions
        all_new_positions = [m['pos'] for m in self.monsters]
        for monster, new_pos in monster_moves:
            # Check if new spot is occupied by another monster or player
            is_occupied = any(p == new_pos for p in all_new_positions) or new_pos == self.player_pos
            if not is_occupied:
                # Update position in the main list for future checks
                idx = all_new_positions.index(monster['pos'])
                all_new_positions[idx] = new_pos
                monster['pos'] = new_pos

        # Check for player damage after all monsters have moved
        for monster in self.monsters:
            if monster['pos'] == self.player_pos:
                self.player_health -= self.MONSTER_DAMAGE
                self.screen_shake = 10 # sfx: player_hit
                self._create_damage_indicator(self.player_pos, self.MONSTER_DAMAGE)

    def _update_game_state(self):
        reward = 0
        terminated = False

        if self.player_health <= 0:
            reward -= 100
            terminated = True
            self.game_over_timer = 30 # Show "Game Over" for 30 steps
            self.player_health = 0
        
        if self.turns_taken >= self.TURN_LIMIT_PER_STAGE and not self.monsters:
            # If time runs out but stage is clear, it's a win
            pass
        elif self.turns_taken >= self.TURN_LIMIT_PER_STAGE:
            reward -= 100
            terminated = True
            self.game_over_timer = 30
        
        if not self.monsters and not terminated:
            if self.stage == self.MAX_STAGES:
                reward += 100
                terminated = True
                self.game_won = True
                self.game_over_timer = 30
            else:
                reward += 10
                self.inter_stage_timer = 30 # Show "Stage Clear" for 30 steps

        return reward, terminated

    def _get_observation(self):
        # Apply screen shake
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
        
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render game elements with shake offset
        self._render_grid(render_offset_x, render_offset_y)
        self._render_potions(render_offset_x, render_offset_y)
        self._render_monsters(render_offset_x, render_offset_y)
        self._render_player(render_offset_x, render_offset_y)
        self._render_particles(render_offset_x, render_offset_y)
        
        # Render UI without shake
        self._render_ui()
        self._render_damage_indicators()

        # Handle message overlays
        if self.inter_stage_timer > 0:
            self._render_message(f"STAGE {self.stage} CLEAR!", f"Next stage in {self.inter_stage_timer // 10+1}...")
        if self.game_over_timer > 0:
            if self.game_won:
                self._render_message("VICTORY!", f"Final Score: {self.score}")
            else:
                self._render_message("GAME OVER", f"Final Score: {self.score}")
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, pos, offset_x=0, offset_y=0):
        gx, gy = pos
        px = self.GRID_TOP_LEFT[0] + gx * self.CELL_SIZE + offset_x
        py = self.GRID_TOP_LEFT[1] + gy * self.CELL_SIZE + offset_y
        return int(px), int(py)

    def _render_grid(self, ox, oy):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_x = self.GRID_TOP_LEFT[0] + i * self.CELL_SIZE + ox
            start_y = self.GRID_TOP_LEFT[1] + oy
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (start_x, start_y + self.GRID_AREA_SIZE), 1)
            # Horizontal lines
            start_x = self.GRID_TOP_LEFT[0] + ox
            start_y = self.GRID_TOP_LEFT[1] + i * self.CELL_SIZE + oy
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (start_x + self.GRID_AREA_SIZE, start_y), 1)

    def _render_player(self, ox, oy):
        px, py = self._grid_to_pixel(self.player_pos, ox, oy)
        s = self.CELL_SIZE
        rect = pygame.Rect(px, py, s, s)
        
        # Draw body with inset
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect.inflate(-s*0.2, -s*0.2))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, rect.inflate(-s*0.5, -s*0.5))

        # Draw facing indicator
        center_x, center_y = px + s // 2, py + s // 2
        fx, fy = self.player_facing
        p1 = (center_x + fx * s * 0.4, center_y + fy * s * 0.4)
        p2 = (center_x - fy * s * 0.2, center_y + fx * s * 0.2)
        p3 = (center_x + fy * s * 0.2, center_y - fx * s * 0.2)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER_ACCENT)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER_ACCENT)

    def _render_monsters(self, ox, oy):
        s = self.CELL_SIZE
        for monster in self.monsters:
            px, py = self._grid_to_pixel(monster['pos'], ox, oy)
            color = (255, 255, 255) if monster['hit_timer'] > 0 else self.COLOR_MONSTER
            accent_color = (255, 255, 255) if monster['hit_timer'] > 0 else self.COLOR_MONSTER_ACCENT
            
            rect = pygame.Rect(px, py, s, s)
            pygame.draw.rect(self.screen, color, rect.inflate(-s*0.2, -s*0.2))
            pygame.draw.rect(self.screen, accent_color, rect.inflate(-s*0.6, -s*0.6))

    def _render_potions(self, ox, oy):
        s = self.CELL_SIZE
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
        for pos in self.potions:
            px, py = self._grid_to_pixel(pos, ox, oy)
            
            # Pulsing glow
            glow_radius = int(s * 0.5 + pulse * 4)
            glow_color = (*self.COLOR_POTION, 50 + pulse * 50)
            pygame.gfxdraw.filled_circle(self.screen, px + s//2, py + s//2, glow_radius, glow_color)

            # Main body
            rect = pygame.Rect(px, py, s, s)
            pygame.draw.rect(self.screen, self.COLOR_POTION, rect.inflate(-s*0.4, -s*0.4))
            pygame.draw.rect(self.screen, self.COLOR_POTION_ACCENT, rect.inflate(-s*0.7, -s*0.7))

    def _render_particles(self, ox, oy):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            size = max(0, p['life'] / p['max_life'] * p['size'])
            px, py = self._grid_to_pixel((p['pos'][0], p['pos'][1]), ox, oy)
            
            # Draw particle as a circle
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(size), p['color'])

    def _create_particles(self, grid_pos, direction, color, count, speed_mult):
        px, py = grid_pos
        for _ in range(count):
            angle = math.atan2(direction[1], direction[0]) + self.np_random.uniform(-0.8, 0.8)
            if direction == (0,0): # Explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
            
            speed = self.np_random.uniform(0.05, 0.15) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [px + 0.5, py + 0.5], 'vel': vel, 'life': life,
                'max_life': life, 'color': color, 'size': self.np_random.uniform(1, 4)
            })

    def _render_damage_indicators(self):
        for ind in self.damage_indicators[:]:
            ind['pos'][1] -= 0.5
            ind['life'] -= 1
            if ind['life'] <= 0:
                self.damage_indicators.remove(ind)
                continue
            
            alpha = min(255, ind['life'] * 20)
            text_surf = self.font_small.render(str(ind['text']), True, ind['color'])
            text_surf.set_alpha(alpha)
            px, py = self._grid_to_pixel((ind['pos'][0], ind['pos'][1]))
            self.screen.blit(text_surf, (px, py))

    def _create_damage_indicator(self, grid_pos, text, color=COLOR_MONSTER):
        self.damage_indicators.append({
            'pos': [grid_pos[0], grid_pos[1]],
            'text': text,
            'life': 30,
            'color': color
        })

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        
        health_color = self.COLOR_HEALTH_LOW
        if health_ratio > 0.66: health_color = self.COLOR_HEALTH_HIGH
        elif health_ratio > 0.33: health_color = self.COLOR_HEALTH_MED
        
        pygame.draw.rect(self.screen, health_color, (10, 10, int(bar_width * health_ratio), bar_height))
        health_text = self.font_small.render(f"HP: {self.player_health}/{self.PLAYER_MAX_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Turn Counter
        turns_left = self.TURN_LIMIT_PER_STAGE - self.turns_taken
        turn_text = self.font_medium.render(f"TURNS: {turns_left}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (self.SCREEN_WIDTH // 2 - turn_text.get_width() // 2, 10))
        
        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, self.SCREEN_HEIGHT - stage_text.get_height() - 5))

    def _render_message(self, main_text, sub_text):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))

        main_surf = self.font_large.render(main_text, True, self.COLOR_TEXT)
        sub_surf = self.font_medium.render(sub_text, True, self.COLOR_TEXT)
        
        main_rect = main_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
        sub_rect = sub_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30))

        s.blit(main_surf, main_rect)
        s.blit(sub_surf, sub_rect)
        self.screen.blit(s, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "turns_taken": self.turns_taken,
            "player_health": self.player_health,
            "monsters_left": len(self.monsters),
        }

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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up pygame window for human play
    pygame.display.set_caption("Grid Combat Arena")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # Map keys to actions for one step
                keys = pygame.key.get_pressed()
                
                # Movement (only one key at a time, prioritized)
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                
                # Other actions
                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

                if terminated:
                    print("--- GAME END ---")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # If auto_advance were True, we'd step here regardless of input
        # Since it's False, we only step on KEYDOWN event.

    env.close()