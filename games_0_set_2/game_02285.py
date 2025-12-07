
# Generated: 2025-08-28T04:21:04.876047
# Source Brief: brief_02285.md
# Brief Index: 2285

        
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
        "Controls: Arrows to move when exploring. In combat: Space for basic attack, "
        "Shift for strong attack, or do nothing to defend. "
        "Move to the glowing door to advance."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore a procedurally generated isometric dungeon, battling enemies in turn-based "
        "combat to reach and defeat the final boss."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_FLOOR = (50, 55, 60)
    COLOR_FLOOR_ACCENT = (60, 65, 70)
    COLOR_WALL = (40, 45, 50)
    COLOR_WALL_TOP = (70, 75, 80)
    
    COLOR_PLAYER = (50, 220, 50)
    COLOR_PLAYER_GLOW = (50, 220, 50, 50)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_BOSS = (255, 100, 100)
    COLOR_DOOR = (255, 220, 100)
    COLOR_DOOR_GLOW = (255, 220, 100, 70)
    
    COLOR_WHITE = (255, 255, 255)
    COLOR_GOLD = (255, 215, 0)
    COLOR_HEALTH_BG = (80, 0, 0)
    COLOR_HEALTH_FG = (0, 180, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_LOG_TEXT = (180, 180, 180)
    
    # Isometric grid
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    GRID_SIZE_X = 10
    GRID_SIZE_Y = 10
    
    # Game mechanics
    MAX_STEPS = 1000
    TOTAL_ROOMS = 5
    PLAYER_MAX_HEALTH = 100
    ENEMY_BASE_HEALTH = 50
    ENEMY_BASE_DAMAGE = 10
    STRONG_ATTACK_MULTIPLIER = 1.8
    DEFEND_DAMAGE_MULTIPLIER = 0.4
    
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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_log = pygame.font.Font(None, 20)
        self.font_title = pygame.font.Font(None, 60)

        self.world_offset = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50)
        
        self.player = {}
        self.enemy = None
        self.particles = []
        self.combat_log = []
        self.room_number = 1
        self.game_state = "EXPLORE" # EXPLORE, COMBAT, TRANSITION, GAMEOVER, VICTORY
        self.door_pos = (0, 0)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = (x - y) * self.TILE_WIDTH_HALF + self.world_offset[0]
        screen_y = (x + y) * self.TILE_HEIGHT_HALF + self.world_offset[1]
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, z_offset, size, height, top_color, side_color):
        """Draws an isometric cube at a grid position."""
        px, py = self._iso_to_screen(x, y)
        py -= z_offset
        
        w, h = self.TILE_WIDTH_HALF * size, self.TILE_HEIGHT_HALF * size
        
        points_top = [ (px, py - h), (px + w, py), (px, py + h), (px - w, py) ]
        points_left = [ (px - w, py), (px, py + h), (px, py + h + height), (px - w, py + height) ]
        points_right = [ (px + w, py), (px, py + h), (px, py + h + height), (px + w, py + height) ]

        pygame.gfxdraw.filled_polygon(surface, points_left, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        
        pygame.gfxdraw.aapolygon(surface, points_left, side_color)
        pygame.gfxdraw.aapolygon(surface, points_right, side_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

    def _generate_room(self):
        """Sets up a new room, spawning an enemy."""
        self.combat_log = ["A new challenge awaits!"]
        self.player['pos'] = (self.GRID_SIZE_X // 2, self.GRID_SIZE_Y - 2)
        
        is_boss_room = self.room_number == self.TOTAL_ROOMS
        
        if is_boss_room:
            enemy_type = 'BOSS'
            health = self.ENEMY_BASE_HEALTH * (1.1 ** (self.room_number - 1)) * 2.5
            damage = self.ENEMY_BASE_DAMAGE * (1.1 ** (self.room_number - 1)) * 1.5
            color = self.COLOR_BOSS
        else:
            enemy_type = 'MINION'
            health = self.ENEMY_BASE_HEALTH * (1.1 ** (self.room_number - 1))
            damage = self.ENEMY_BASE_DAMAGE * (1.1 ** (self.room_number - 1))
            color = self.COLOR_ENEMY

        self.enemy = {
            'pos': (self.GRID_SIZE_X // 2, 2),
            'health': health,
            'max_health': health,
            'damage': damage,
            'type': enemy_type,
            'color': color,
            'flash': 0,
        }
        self.game_state = "COMBAT"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.room_number = 1
        
        self.player = {
            'pos': (self.GRID_SIZE_X // 2, self.GRID_SIZE_Y - 2),
            'health': self.PLAYER_MAX_HEALTH,
            'max_health': self.PLAYER_MAX_HEALTH,
            'gold': 0,
            'flash': 0,
            'is_defending': False,
        }
        
        self.enemy = None
        self.particles = []
        
        self._generate_room()
        
        return self._get_observation(), self._get_info()

    def _add_log(self, message):
        self.combat_log.insert(0, message)
        if len(self.combat_log) > 4:
            self.combat_log.pop()

    def _create_particles(self, pos, color, count):
        px, py = self._iso_to_screen(*pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Turn ---
        if self.game_state == "COMBAT":
            self.player['is_defending'] = False
            player_action = "DEFEND" # Default action
            
            if shift_held:
                player_action = "STRONG_ATTACK"
            elif space_held:
                player_action = "BASIC_ATTACK"
            
            if player_action == "BASIC_ATTACK":
                damage = self.np_random.integers(15, 26)
                self.enemy['health'] -= damage
                self.enemy['flash'] = 3
                reward += 0.1
                self._add_log(f"Player strikes for {damage} damage!")
                self._create_particles(self.enemy['pos'], self.COLOR_WHITE, 15)
                # Sound: player_attack.wav

            elif player_action == "STRONG_ATTACK":
                damage = int(self.np_random.integers(15, 26) * self.STRONG_ATTACK_MULTIPLIER)
                self.enemy['health'] -= damage
                self.enemy['flash'] = 5
                reward += 0.1
                self._add_log(f"Player unleashes a strong hit for {damage}!")
                self._create_particles(self.enemy['pos'], self.COLOR_GOLD, 30)
                # Sound: strong_attack.wav
                
            elif player_action == "DEFEND":
                self.player['is_defending'] = True
                self._add_log("Player braces for an attack.")
                # Sound: defend.wav
        
        elif self.game_state == "EXPLORE":
            if movement != 0:
                old_pos = self.player['pos']
                new_pos = list(old_pos)
                if movement == 1: new_pos[1] -= 1 # Up
                elif movement == 2: new_pos[1] += 1 # Down
                elif movement == 3: new_pos[0] -= 1 # Left
                elif movement == 4: new_pos[0] += 1 # Right
                
                # Clamp to grid
                new_pos[0] = max(0, min(self.GRID_SIZE_X - 1, new_pos[0]))
                new_pos[1] = max(0, min(self.GRID_SIZE_Y - 1, new_pos[1]))
                self.player['pos'] = tuple(new_pos)

                if self.player['pos'] == self.door_pos:
                    self.game_state = "TRANSITION"
                    # Sound: door_open.wav
        
        # --- Enemy Turn & State Update ---
        if self.game_state == "COMBAT" and self.enemy['health'] > 0:
            # Enemy AI
            damage = int(self.np_random.uniform(0.8, 1.2) * self.enemy['damage'])
            if self.player['is_defending']:
                damage = int(damage * self.DEFEND_DAMAGE_MULTIPLIER)
                self._add_log(f"Enemy attacks! Player defends, taking {damage} damage.")
            else:
                self._add_log(f"Enemy attacks for {damage} damage!")
            
            self.player['health'] -= damage
            self.player['flash'] = 3
            reward -= 0.1
            self._create_particles(self.player['pos'], self.COLOR_PLAYER, 15)
            # Sound: player_hit.wav
            
            if self.player['health'] <= 0:
                self.player['health'] = 0
                self.game_state = "GAMEOVER"
                self._add_log("Player has been defeated.")
                # Sound: game_over.wav
        
        # Check for enemy defeat
        if self.enemy and self.enemy['health'] <= 0:
            # Sound: enemy_defeat.wav
            if self.enemy['type'] == 'BOSS':
                reward += 100
                self.score += 100
                self.player['gold'] += 500
                self.game_state = "VICTORY"
                self._add_log("The Final Boss is vanquished! Victory!")
                # Sound: victory.wav
            else:
                reward += 10
                self.score += 10
                gold_gain = self.np_random.integers(10, 21) * self.room_number
                self.player['gold'] += gold_gain
                self._add_log(f"Enemy defeated! Gained {gold_gain} gold.")
                self.game_state = "EXPLORE"
                self.door_pos = self.enemy['pos']
                self.enemy = None
        
        if self.game_state == "TRANSITION":
            self.room_number += 1
            self._generate_room()
            
        # Update flash timers
        if self.player['flash'] > 0: self.player['flash'] -= 1
        if self.enemy and self.enemy['flash'] > 0: self.enemy['flash'] -= 1
            
        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1

        self.steps += 1
        if self.steps >= self.MAX_STEPS or self.game_state in ["GAMEOVER", "VICTORY"]:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor
        floor_points = [
            self._iso_to_screen(0, 0),
            self._iso_to_screen(self.GRID_SIZE_X, 0),
            self._iso_to_screen(self.GRID_SIZE_X, self.GRID_SIZE_Y),
            self._iso_to_screen(0, self.GRID_SIZE_Y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)
        
        # Draw grid lines for texture
        for i in range(self.GRID_SIZE_X + 1):
            p1 = self._iso_to_screen(i, 0)
            p2 = self._iso_to_screen(i, self.GRID_SIZE_Y)
            pygame.draw.line(self.screen, self.COLOR_FLOOR_ACCENT, p1, p2, 1)
        for i in range(self.GRID_SIZE_Y + 1):
            p1 = self._iso_to_screen(0, i)
            p2 = self._iso_to_screen(self.GRID_SIZE_X, i)
            pygame.draw.line(self.screen, self.COLOR_FLOOR_ACCENT, p1, p2, 1)
        
        pygame.gfxdraw.aapolygon(self.screen, floor_points, self.COLOR_FLOOR_ACCENT)

        # --- Draw entities in Z-order ---
        entities = []
        if self.enemy: entities.append(self.enemy)
        entities.append(self.player)
        
        # Add door if in explore state
        if self.game_state == "EXPLORE" and self.enemy is None:
            entities.append({'type': 'DOOR', 'pos': self.door_pos})

        # Sort by grid y-position for correct occlusion
        sorted_entities = sorted(entities, key=lambda e: e['pos'][1])

        for entity in sorted_entities:
            x, y = entity['pos']
            
            if entity.get('type') == 'DOOR':
                px, py = self._iso_to_screen(x, y)
                py -= 10
                pygame.gfxdraw.filled_circle(self.screen, px, py, 15, self.COLOR_DOOR_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 10, self.COLOR_DOOR)
                pygame.gfxdraw.aacircle(self.screen, px, py, 10, self.COLOR_WHITE)
                continue

            color = self.COLOR_PLAYER if entity == self.player else entity['color']
            if entity.get('flash', 0) > 0:
                color = self.COLOR_WHITE
            
            # Shadow
            shadow_px, shadow_py = self._iso_to_screen(x, y)
            shadow_rect = pygame.Rect(0, 0, 30, 15)
            shadow_rect.center = (shadow_px, shadow_py + 15)
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (0, 0, 0, 80), (0, 0, *shadow_rect.size))
            self.screen.blit(shadow_surf, shadow_rect.topleft)

            # Glow for player
            if entity == self.player:
                glow_px, glow_py = self._iso_to_screen(x, y)
                glow_py -= 10
                pygame.gfxdraw.filled_circle(self.screen, glow_px, glow_py, 20, self.COLOR_PLAYER_GLOW)

            # Character cube
            self._draw_iso_cube(self.screen, x, y, 10, 0.7, 20, color, tuple(c*0.7 for c in color))
            
            # Health bar
            if 'health' in entity and 'max_health' in entity:
                px, py = self._iso_to_screen(x, y)
                bar_width = 40
                bar_height = 5
                health_pct = max(0, entity['health'] / entity['max_health'])
                bg_rect = pygame.Rect(px - bar_width // 2, py - 45, bar_width, bar_height)
                fg_rect = pygame.Rect(px - bar_width // 2, py - 45, int(bar_width * health_pct), bar_height)
                
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG if entity == self.player else self.COLOR_ENEMY, fg_rect)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, bg_rect, 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (3, 3), 3)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - 3, int(p['pos'][1]) - 3), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Player Stats
        health_text = self.font_ui.render(f"HP: {int(self.player['health'])} / {self.player['max_health']}", True, self.COLOR_UI_TEXT)
        gold_text = self.font_ui.render(f"Gold: {self.player['gold']}", True, self.COLOR_GOLD)
        room_text = self.font_ui.render(f"Dungeon Level: {self.room_number} / {self.TOTAL_ROOMS}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(gold_text, (10, 35))
        self.screen.blit(room_text, (self.SCREEN_WIDTH - room_text.get_width() - 10, 10))
        
        # Combat Log
        log_y = self.SCREEN_HEIGHT - 25
        for i, msg in enumerate(self.combat_log):
            alpha = 255 - (i * 60)
            log_surf = self.font_log.render(msg, True, (*self.COLOR_LOG_TEXT, alpha))
            log_surf.set_alpha(alpha)
            self.screen.blit(log_surf, (10, log_y - i * 20))
            
        # Game Over / Victory Message
        if self.game_state in ["GAMEOVER", "VICTORY"]:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "VICTORY" if self.game_state == "VICTORY" else "GAME OVER"
            color = self.COLOR_GOLD if self.game_state == "VICTORY" else self.COLOR_ENEMY
            
            title_surf = self.font_title.render(message, True, color)
            title_rect = title_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            overlay.blit(title_surf, title_rect)
            self.screen.blit(overlay, (0, 0))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player['health'],
            "player_gold": self.player['gold'],
            "room_number": self.room_number,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Dungeon Crawler")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # None
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Manual reset

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # In a real-time game, you'd step every frame.
        # Since this is turn-based (auto_advance=False), we only step when an action is taken.
        # For manual play, we can simulate this by stepping on any key press.
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                # Optional: wait for a moment before auto-resetting
                pygame.time.wait(3000)
                obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for manual play

    env.close()