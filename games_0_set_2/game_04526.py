
# Generated: 2025-08-28T02:40:07.594968
# Source Brief: brief_04526.md
# Brief Index: 4526

        
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

    user_guide = (
        "Controls: Arrow keys to move. Space for a basic attack, Shift for a power attack (double damage when adjacent)."
    )

    game_description = (
        "Explore a dangerous dungeon, battle monsters, and gain experience to reach level 5 in this turn-based isometric RPG."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 15, 15
        self.MAX_STEPS = 1000
        self.TILE_W, self.TILE_H = 32, 16
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 80

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (40, 40, 55)
        self.COLOR_FLOOR_ALT = (45, 45, 60)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_PLAYER_GLOW = (100, 255, 100, 50)
        self.COLOR_ENEMY_WEAK = (200, 50, 50)
        self.COLOR_ENEMY_MEDIUM = (220, 100, 50)
        self.COLOR_ENEMY_STRONG = (255, 150, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_HEALTH_FG = (200, 0, 0)
        self.COLOR_XP_BG = (20, 40, 80)
        self.COLOR_XP_FG = (50, 120, 255)
        self.COLOR_LOG_BG = (0, 0, 0, 128)

        # Enemy types
        self.ENEMY_TYPES = {
            "weak": {"hp": 10, "atk": 5, "xp": 10, "color": self.COLOR_ENEMY_WEAK},
            "medium": {"hp": 20, "atk": 10, "xp": 25, "color": self.COLOR_ENEMY_MEDIUM},
            "strong": {"hp": 30, "atk": 15, "xp": 50, "color": self.COLOR_ENEMY_STRONG},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.player = {}
        self.enemies = []
        self.particles = []
        self.log_messages = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {
            "pos": [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2],
            "level": 1,
            "xp": 0,
            "xp_to_next": 100,
            "max_health": 100,
            "health": 100,
        }
        self.enemies = []
        self.particles = []
        self.log_messages = ["Welcome to the dungeon!", "Reach level 5 to win."]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        did_attack = False
        action_performed = "Waited."

        # 1. Player Action
        if space_held or shift_held:
            target, dist = self._find_nearest_enemy()
            if target:
                did_attack = True
                is_power_attack = shift_held
                is_adjacent = dist <= 1.5 # sqrt(1^2+1^2) is ~1.41
                
                damage = 10
                if is_power_attack:
                    damage *= 2 if is_adjacent else 1
                    action_performed = f"Power attack! ({'Adjacent' if is_adjacent else 'Ranged'})"
                else:
                    action_performed = "Basic attack."

                target['health'] -= damage
                reward += damage * 0.1 # Scaled reward for damage
                self.score += damage * 0.1
                self._add_particle(target['pos'], (255, 255, 0), 15, 10)
                # Sfx: player_attack.wav
                
                if target['health'] <= 0:
                    action_performed += f" Defeated a {target['type']}!"
                    reward += 10
                    self.score += 10
                    self._gain_xp(target['xp'])
                    self.enemies.remove(target)
                    # Sfx: enemy_die.wav
            else:
                action_performed = "Attacked, but no enemies are near."

        elif movement > 0:
            px, py = self.player['pos']
            if movement == 1: py -= 1 # Up
            elif movement == 2: py += 1 # Down
            elif movement == 3: px -= 1 # Left
            elif movement == 4: px += 1 # Right
            
            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                self.player['pos'] = [px, py]
                action_performed = "Moved."
                # Sfx: player_move.wav
            else:
                action_performed = "Tried to move into a wall."

        if not did_attack:
            reward -= 0.2
            self.score -= 0.2

        self._add_log(action_performed)

        # 2. Enemy Turn
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - self.player['pos'][0], enemy['pos'][1] - self.player['pos'][1])
            if dist <= 1.5:
                damage = self.ENEMY_TYPES[enemy['type']]['atk']
                self.player['health'] -= damage
                self._add_log(f"Took {damage} damage from a {enemy['type']}.")
                self._add_particle(self.player['pos'], (255, 0, 0), 20, 15)
                # Sfx: player_hurt.wav
        
        # 3. Game State Updates
        self._update_particles()
        self._spawn_enemy()
        
        # 4. Termination Check
        terminated = False
        if self.player['health'] <= 0:
            terminated = True
            reward -= 100
            self.score -= 100
            self.game_over = True
            self._add_log("You have been defeated. Game Over.")
            # Sfx: game_over.wav
        elif self.player['level'] >= 5:
            terminated = True
            reward += 100
            self.score += 100
            self.game_over = True
            self._add_log("Congratulations! You reached level 5!")
            # Sfx: victory.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self._add_log("You ran out of time. Game Over.")
        
        # Ensure health doesn't exceed max
        self.player['health'] = min(self.player['health'], self.player['max_health'])
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _gain_xp(self, amount):
        self.player['xp'] += amount
        self._add_log(f"Gained {amount} XP.")
        while self.player['xp'] >= self.player['xp_to_next']:
            self._level_up()

    def _level_up(self):
        reward = 20
        self.score += 20
        self.player['level'] += 1
        self.player['xp'] -= self.player['xp_to_next']
        self.player['xp_to_next'] = int(self.player['xp_to_next'] * 1.5)
        self.player['max_health'] += 20
        self.player['health'] = self.player['max_health'] # Full heal on level up
        self._add_log(f"Leveled up to level {self.player['level']}! Health restored.")
        # Sfx: level_up.wav

    def _spawn_enemy(self):
        spawn_chance = 0.1 * (1 + 0.1 * (self.player['level'] - 1))
        if self.np_random.random() < spawn_chance and len(self.enemies) < 10:
            for _ in range(10): # Try 10 times to find an empty spot
                x = self.np_random.integers(0, self.GRID_WIDTH)
                y = self.np_random.integers(0, self.GRID_HEIGHT)
                
                dist_to_player = math.hypot(x - self.player['pos'][0], y - self.player['pos'][1])
                if dist_to_player < 3: continue # Don't spawn too close

                is_occupied = any(e['pos'] == [x, y] for e in self.enemies)
                if not is_occupied:
                    enemy_type_key = self.np_random.choice(list(self.ENEMY_TYPES.keys()))
                    enemy_type = self.ENEMY_TYPES[enemy_type_key]
                    self.enemies.append({
                        "pos": [x, y],
                        "type": enemy_type_key,
                        "max_health": enemy_type['hp'],
                        "health": enemy_type['hp'],
                        "xp": enemy_type['xp'],
                    })
                    self._add_log(f"A wild {enemy_type_key} appears!")
                    # Sfx: enemy_spawn.wav
                    break

    def _find_nearest_enemy(self):
        if not self.enemies:
            return None, float('inf')
        
        px, py = self.player['pos']
        closest_enemy = min(
            self.enemies,
            key=lambda e: math.hypot(e['pos'][0] - px, e['pos'][1] - py)
        )
        dist = math.hypot(closest_enemy['pos'][0] - px, closest_enemy['pos'][1] - py)
        return closest_enemy, dist

    def _add_particle(self, pos, color, radius, life):
        self.particles.append({
            'pos': list(pos),
            'color': color,
            'max_radius': radius,
            'radius': 0,
            'max_life': life,
            'life': life,
        })
    
    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            p['radius'] = p['max_radius'] * (1 - (p['life'] / p['max_life']))
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _add_log(self, message):
        self.log_messages.append(message)
        if len(self.log_messages) > 3:
            self.log_messages.pop(0)

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
            "level": self.player['level'],
            "health": self.player['health'],
            "xp": self.player['xp'],
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, x, y, color, height_offset=0):
        sx, sy = self._iso_to_screen(x, y)
        sy -= height_offset
        points = [
            (sx, sy - self.TILE_H / 2),
            (sx + self.TILE_W / 2, sy),
            (sx, sy + self.TILE_H / 2),
            (sx - self.TILE_W / 2, sy),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_bar(self, surface, x, y, width, height, val, max_val, fg, bg):
        if max_val == 0: return
        pygame.draw.rect(surface, bg, (x, y, width, height))
        fill_width = int(width * (val / max_val))
        if fill_width > 0:
            pygame.draw.rect(surface, fg, (x, y, fill_width, height))
    
    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "left":
            text_rect.midleft = pos
        elif align == "right":
            text_rect.midright = pos
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        # Draw floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.COLOR_FLOOR if (x + y) % 2 == 0 else self.COLOR_FLOOR_ALT
                self._draw_iso_tile(self.screen, x, y, color)
        
        # Draw entities sorted by y-pos for correct layering
        entities = [{'obj': self.player, 'type': 'player'}] + [{'obj': e, 'type': 'enemy'} for e in self.enemies]
        entities.sort(key=lambda e: e['obj']['pos'][0] + e['obj']['pos'][1])

        for entity_info in entities:
            obj = entity_info['obj']
            sx, sy = self._iso_to_screen(obj['pos'][0], obj['pos'][1])
            
            if entity_info['type'] == 'player':
                # Player Glow
                glow_points = [
                    (sx, sy - self.TILE_H),
                    (sx + self.TILE_W, sy - self.TILE_H / 2),
                    (sx, sy),
                    (sx - self.TILE_W, sy - self.TILE_H / 2),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
                # Player Body
                self._draw_iso_tile(self.screen, obj['pos'][0], obj['pos'][1], self.COLOR_PLAYER, height_offset=self.TILE_H / 2)
            else: # Enemy
                color = self.ENEMY_TYPES[obj['type']]['color']
                self._draw_iso_tile(self.screen, obj['pos'][0], obj['pos'][1], color, height_offset=self.TILE_H / 2)
                # Enemy Health Bar
                bar_width = 30
                self._draw_bar(self.screen, sx - bar_width/2, sy - self.TILE_H - 10, bar_width, 5, obj['health'], obj['max_health'], self.COLOR_HEALTH_FG, self.COLOR_HEALTH_BG)

        # Draw particles on top
        for p in self.particles:
            sx, sy = self._iso_to_screen(p['pos'][0], p['pos'][1])
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.aacircle(self.screen, sx, int(sy - self.TILE_H/2), int(p['radius']), color)

    def _render_ui(self):
        # UI Panel
        panel_height = 60
        pygame.draw.rect(self.screen, (10, 10, 15), (0, 0, self.WIDTH, panel_height))
        pygame.draw.line(self.screen, (80, 80, 100), (0, panel_height), (self.WIDTH, panel_height))

        # Health Bar
        self._draw_text("HP", (15, 20), self.font_medium, self.COLOR_TEXT, "left")
        self._draw_bar(self.screen, 50, 12, 200, 16, self.player['health'], self.player['max_health'], self.COLOR_PLAYER, self.COLOR_HEALTH_BG)
        hp_text = f"{int(self.player['health'])} / {int(self.player['max_health'])}"
        self._draw_text(hp_text, (150, 20), self.font_medium, self.COLOR_TEXT)

        # XP Bar
        self._draw_text("XP", (15, 45), self.font_medium, self.COLOR_TEXT, "left")
        self._draw_bar(self.screen, 50, 37, 200, 16, self.player['xp'], self.player['xp_to_next'], self.COLOR_XP_FG, self.COLOR_XP_BG)
        xp_text = f"{int(self.player['xp'])} / {int(self.player['xp_to_next'])}"
        self._draw_text(xp_text, (150, 45), self.font_medium, self.COLOR_TEXT)

        # Level display
        level_text = f"LEVEL {self.player['level']}"
        self._draw_text(level_text, (290, 30), self.font_large, (255, 215, 0))

        # Score and Steps
        self._draw_text(f"Score: {int(self.score)}", (self.WIDTH - 10, 20), self.font_medium, self.COLOR_TEXT, "right")
        self._draw_text(f"Turn: {self.steps}/{self.MAX_STEPS}", (self.WIDTH - 10, 45), self.font_medium, self.COLOR_TEXT, "right")

        # Log Messages
        log_surface = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        log_surface.fill(self.COLOR_LOG_BG)
        for i, msg in enumerate(self.log_messages):
            self._draw_text(msg, (10, 10 + i * 18), self.font_small, self.COLOR_TEXT, "left")
        self.screen.blit(log_surface, (0, self.HEIGHT - 60))
        
        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "YOU WIN!" if self.player['level'] >= 5 else "GAME OVER"
            self._draw_text(msg, (self.WIDTH/2, self.HEIGHT/2), self.font_large, (255,255,255))


    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium environments are not designed for direct keyboard play without a wrapper.
    # This is a simplified loop for testing and visualization.
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for pygame.display to work headlessly
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a visible window for playing
    pygame.display.set_caption("Isometric Dungeon Crawler")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    print("\n" + "="*30)
    print("      MANUAL PLAY TESTER")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Close the window to quit.")
    print("="*30 + "\n")


    while not terminated:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                # This is a one-shot key press, not a hold.
                # For this simple test, we treat it as the action for this turn.
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                if keys[pygame.K_SPACE]: space = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

                action = [movement, space, shift]
                obs, reward, term, trunc, info = env.step(action)
                terminated = term
                
                print(f"Turn: {info['steps']}, Lvl: {info['level']}, HP: {info['health']}, Score: {info['score']:.1f}, Reward: {reward:.1f}")


        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if env.game_over:
            pygame.time.wait(3000) # Pause on game over screen
            break

    pygame.quit()