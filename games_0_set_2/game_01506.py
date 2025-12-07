
# Generated: 2025-08-27T17:21:45.336944
# Source Brief: brief_01506.md
# Brief Index: 1506

        
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
    """
    An isometric arcade game where the player battles monsters using strategic abilities.
    The goal is to defeat all monsters before the player's health runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift for an area-of-effect blast. "
        "Press Space to fire a projectile at the nearest monster."
    )
    game_description = (
        "Mash monsters in an isometric 2D arcade environment. "
        "Use strategic abilities to maximize your score before you are overwhelmed."
    )

    # Frame advance behavior
    auto_advance = False

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_SHADOW = (30, 150, 90)
    
    # Monster Colors
    COLOR_MONSTER_BASIC = (255, 80, 80)
    COLOR_MONSTER_ARMORED = (200, 70, 70)
    COLOR_MONSTER_SPEEDY = (255, 120, 120)
    COLOR_MONSTER_SHADOW = (150, 50, 50)

    # UI & Effects Colors
    COLOR_HEALTH_BAR_BG = (10, 10, 10)
    COLOR_HEALTH_BAR_PLAYER = (50, 255, 150)
    COLOR_HEALTH_BAR_MONSTER = (255, 80, 80)
    COLOR_PROJECTILE = (100, 180, 255)
    COLOR_EXPLOSION = (255, 220, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 220, 50)
    
    # --- Game Constants ---
    GRID_WIDTH = 18
    GRID_HEIGHT = 12
    TILE_WIDTH = 50
    TILE_HEIGHT = 25
    MAX_STEPS = 1000
    NUM_MONSTERS = 15
    PLAYER_MAX_HEALTH = 100
    MONSTER_ATTACK_DAMAGE = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 36)
        self.font_damage = pygame.font.Font(None, 22)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.monsters = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.effects = None
        self.screen_shake = 0

        # Isometric projection offsets
        self.iso_offset_x = self.screen.get_width() // 2
        self.iso_offset_y = 100
        
        # Initialize state
        self.reset()

        # Validate implementation after setup
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.effects = []

        # Generate monsters
        self.monsters = []
        occupied_tiles = {tuple(self.player_pos)}
        for _ in range(self.NUM_MONSTERS):
            # Find a valid spawn location
            while True:
                pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
                dist_to_player = math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1])
                if tuple(pos) not in occupied_tiles and dist_to_player >= 3:
                    occupied_tiles.add(tuple(pos))
                    break
            
            # Choose monster type
            monster_type_roll = self.np_random.random()
            if monster_type_roll < 0.6:
                m_type = "basic"
                hp = 10
            elif monster_type_roll < 0.85:
                m_type = "armored"
                hp = 20
            else:
                m_type = "speedy"
                hp = 5
            
            self.monsters.append({
                "pos": pos,
                "type": m_type,
                "max_hp": hp,
                "hp": hp,
                "id": self.np_random.integers(1, 1_000_000) # unique id for dict key
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Per-turn penalty to encourage efficiency
        self.steps += 1
        self.effects.clear()

        # --- Player Action Phase ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Abilities (prioritize shift)
        if shift_held:
            # AoE Explosion
            # sound: player_explosion.wav
            self._add_effect('explosion', self.player_pos, 2.5, self.COLOR_EXPLOSION, 15)
            monsters_hit = 0
            for m in self.monsters:
                dist = math.hypot(m['pos'][0] - self.player_pos[0], m['pos'][1] - self.player_pos[1])
                if dist <= 2:
                    damage = 5
                    reward += self._apply_damage(m, damage)
                    monsters_hit += 1
            if monsters_hit * 5 > 5:
                reward += 2.0

        elif space_held:
            # Projectile Attack
            target = self._find_nearest_monster()
            if target:
                # sound: player_projectile.wav
                self._add_effect('projectile', self.player_pos, 1, self.COLOR_PROJECTILE, 10, target=target['pos'])
                damage = 5 if target['type'] == 'armored' else 10
                reward += self._apply_damage(target, damage)
                if damage > 5:
                    reward += 2.0

        # 2. Handle Movement
        else: # No ability used, can move
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right

            if dx != 0 or dy != 0:
                new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
                if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                    self.player_pos = new_pos
        
        # --- Monster Action Phase ---
        monsters_to_remove = [m for m in self.monsters if m['hp'] <= 0]
        for dead_monster in monsters_to_remove:
            self.monsters.remove(dead_monster)

        for m in self.monsters:
            # Speedy monster movement
            if m['type'] == 'speedy' and self.np_random.random() < 0.5:
                # Simple random move, could be improved to be smarter
                move_options = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random_move = self.np_random.choice(len(move_options))
                dx, dy = move_options[random_move]
                new_m_pos = [m['pos'][0] + dx, m['pos'][1] + dy]
                if 0 <= new_m_pos[0] < self.GRID_WIDTH and 0 <= new_m_pos[1] < self.GRID_HEIGHT:
                    m['pos'] = new_m_pos

            # Monster attack
            dist_to_player = math.hypot(m['pos'][0] - self.player_pos[0], m['pos'][1] - self.player_pos[1])
            if dist_to_player < 1.5: # Adjacency check
                # sound: player_hit.wav
                self.player_health -= self.MONSTER_ATTACK_DAMAGE
                reward -= 5.0
                self.screen_shake = 10
                self._add_effect('damage_text', [30, 60], 1, self.COLOR_MONSTER_BASIC, 20, text=f"-{self.MONSTER_ATTACK_DAMAGE}")

        # --- Termination Check ---
        terminated = False
        if self.player_health <= 0:
            self.player_health = 0
            reward -= 50
            terminated = True
            self.game_over = True
        elif not self.monsters:
            reward += 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _apply_damage(self, monster, damage):
        """Applies damage to a monster and returns the reward."""
        actual_damage = min(monster['hp'], damage)
        monster['hp'] -= actual_damage
        reward = 0.2  # Reward for any damage
        self._add_effect('damage_text', monster['pos'], 1, self.COLOR_TEXT, 20, text=f"{actual_damage}")
        
        if monster['hp'] <= 0:
            # sound: monster_die.wav
            kill_bonuses = {"basic": 1.0, "armored": 2.0, "speedy": 0.5}
            reward += kill_bonuses[monster['type']]
            self.score += int(kill_bonuses[monster['type']] * 100)
            self._add_effect('explosion', monster['pos'], 1.5, self.COLOR_MONSTER_SHADOW, 10)
        return reward

    def _find_nearest_monster(self):
        """Finds the monster closest to the player."""
        if not self.monsters:
            return None
        
        nearest_monster = min(
            self.monsters,
            key=lambda m: math.hypot(m['pos'][0] - self.player_pos[0], m['pos'][1] - self.player_pos[1])
        )
        return nearest_monster

    def _grid_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = self.iso_offset_x + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.iso_offset_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _add_effect(self, type, pos, size, color, duration, **kwargs):
        self.effects.append({
            'type': type, 'pos': pos, 'size': size, 'color': color,
            'duration': duration, 'max_duration': duration, **kwargs
        })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "player_health": self.player_health}

    def _get_observation(self):
        # Apply screen shake
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            render_offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            self.screen_shake = int(self.screen_shake * 0.8) # decay shake

        # Create a temporary surface to draw on, then blit to main screen with offset
        temp_surface = self.screen.copy()
        temp_surface.fill(self.COLOR_BG)

        # Render all game elements to the temporary surface
        self._render_grid(temp_surface)
        self._render_shadows(temp_surface)
        self._render_monsters(temp_surface)
        self._render_player(temp_surface)
        self._render_effects(temp_surface)
        self._render_ui(temp_surface)
        
        if self.game_over:
            self._render_game_over(temp_surface)

        # Blit the temp surface to the main screen with the shake offset
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(temp_surface, (render_offset_x, render_offset_y))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self, surface):
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = self._grid_to_screen(-0.5, y - 0.5)
            end_pos = self._grid_to_screen(self.GRID_WIDTH - 0.5, y - 0.5)
            pygame.draw.aaline(surface, self.COLOR_GRID, start_pos, end_pos)
        for x in range(self.GRID_WIDTH + 1):
            start_pos = self._grid_to_screen(x - 0.5, -0.5)
            end_pos = self._grid_to_screen(x - 0.5, self.GRID_HEIGHT - 0.5)
            pygame.draw.aaline(surface, self.COLOR_GRID, start_pos, end_pos)

    def _render_shadows(self, surface):
        # Player shadow
        px, py = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        pygame.gfxdraw.filled_ellipse(surface, px, py + 12, 12, 6, self.COLOR_PLAYER_SHADOW)
        
        # Monster shadows
        for m in self.monsters:
            mx, my = self._grid_to_screen(m['pos'][0], m['pos'][1])
            pygame.gfxdraw.filled_ellipse(surface, mx, my + 10, 10, 5, self.COLOR_MONSTER_SHADOW)

    def _render_player(self, surface):
        x, y = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        # Simple bobbing animation
        bob = int(math.sin(pygame.time.get_ticks() * 0.005) * 3)
        y += bob
        
        # Draw player as a rectangle
        player_rect = pygame.Rect(0, 0, 20, 30)
        player_rect.center = (x, y)
        pygame.draw.rect(surface, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Health bar
        self._render_health_bar(surface, (x, y - 25), self.player_health, self.PLAYER_MAX_HEALTH, self.COLOR_HEALTH_BAR_PLAYER)

    def _render_monsters(self, surface):
        for m in self.monsters:
            x, y = self._grid_to_screen(m['pos'][0], m['pos'][1])
            
            # Different shapes for monster types
            color = self.COLOR_MONSTER_BASIC
            if m['type'] == 'armored':
                color = self.COLOR_MONSTER_ARMORED
                monster_rect = pygame.Rect(0, 0, 22, 22)
                monster_rect.center = (x, y)
                pygame.draw.rect(surface, color, monster_rect, border_radius=6)
            elif m['type'] == 'speedy':
                color = self.COLOR_MONSTER_SPEEDY
                pygame.gfxdraw.filled_trigon(surface, x, y - 10, x - 10, y + 8, x + 10, y + 8, color)
            else: # Basic
                pygame.gfxdraw.filled_circle(surface, x, y, 10, color)

            # Health bar
            self._render_health_bar(surface, (x, y - 20), m['hp'], m['max_hp'], self.COLOR_HEALTH_BAR_MONSTER)

    def _render_health_bar(self, surface, pos, current_hp, max_hp, color):
        x, y = pos
        width, height = 30, 5
        bg_rect = pygame.Rect(x - width // 2, y, width, height)
        
        health_ratio = max(0, current_hp / max_hp)
        fg_width = int(width * health_ratio)
        fg_rect = pygame.Rect(x - width // 2, y, fg_width, height)

        pygame.draw.rect(surface, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=2)
        if fg_width > 0:
            pygame.draw.rect(surface, color, fg_rect, border_radius=2)
            
    def _render_effects(self, surface):
        for effect in self.effects:
            progress = effect['duration'] / effect['max_duration']
            
            if effect['type'] == 'explosion':
                x, y = self._grid_to_screen(effect['pos'][0], effect['pos'][1])
                radius = int(effect['size'] * self.TILE_WIDTH/2 * (1 - progress))
                alpha = int(255 * progress)
                color = (*effect['color'], alpha)
                if radius > 0:
                    pygame.gfxdraw.aacircle(surface, x, y, radius, color)
                    pygame.gfxdraw.aacircle(surface, x, y, radius-1, color)
            
            elif effect['type'] == 'projectile':
                start_x, start_y = self._grid_to_screen(effect['pos'][0], effect['pos'][1])
                end_x, end_y = self._grid_to_screen(effect['target'][0], effect['target'][1])
                pygame.draw.aaline(surface, effect['color'], (start_x, start_y-10), (end_x, end_y-10), 2)
                
            elif effect['type'] == 'damage_text':
                if isinstance(effect['pos'], list) and len(effect['pos']) == 2 and not isinstance(effect['pos'][0], list):
                    # Grid position
                    x, y = self._grid_to_screen(effect['pos'][0], effect['pos'][1])
                    y -= int(20 * (1 - progress)) # Move text up
                else:
                    # Screen position
                    x, y = effect['pos']
                
                alpha = int(255 * progress)
                text_surf = self.font_damage.render(effect['text'], True, effect['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=(x, y))
                surface.blit(text_surf, text_rect)
            
            effect['duration'] -= 1

    def _render_ui(self, surface):
        # Player Health UI
        health_text = self.font_small.render("HEALTH", True, self.COLOR_TEXT)
        surface.blit(health_text, (20, 15))
        self._render_health_bar(surface, (75, 40), self.player_health, self.PLAYER_MAX_HEALTH, self.COLOR_HEALTH_BAR_PLAYER)
        pygame.draw.rect(surface, self.COLOR_TEXT, (20, 38, 110, 9), 1, border_radius=3)
        
        # Score UI
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(topright=(self.screen.get_width() - 20, 15))
        surface.blit(score_text, score_rect)

    def _render_game_over(self, surface):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        surface.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if not self.monsters else "GAME OVER"
        color = self.COLOR_PLAYER if not self.monsters else self.COLOR_MONSTER_BASIC
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2 - 20))
        surface.blit(text, text_rect)

        final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        final_score_rect = final_score_text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2 + 20))
        surface.blit(final_score_text, final_score_rect)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # For headless validation
    
    env = GameEnv()
    
    # Run a headless test
    print("--- Headless Test ---")
    obs, info = env.reset()
    print(f"Initial Info: {info}")
    terminated = False
    total_reward = 0
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {_ + 1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            break
    print(f"Total Reward after 5 random steps: {total_reward:.2f}")
    env.close()

    # To run with visualization, you would need a display
    # Example for local execution with a window:
    #
    # if 'SDL_VIDEODRIVER' in os.environ:
    #     del os.environ['SDL_VIDEODRIVER']
    #
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((640, 400))
    # pygame.display.set_caption("Monster Masher")
    # clock = pygame.time.Clock()
    #
    # obs, info = env.reset()
    # running = True
    # while running:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
    #
    #     if keys[pygame.K_SPACE]: space = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
    #
    #     action = [movement, space, shift]
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         pygame.time.wait(3000)
    #         obs, info = env.reset()
    #
    #     # Since auto_advance is False, we need a small delay to make it playable by humans
    #     clock.tick(10) # 10 FPS for human playability
    #
    # env.close()