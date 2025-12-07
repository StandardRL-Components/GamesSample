
# Generated: 2025-08-28T05:48:06.014945
# Source Brief: brief_05698.md
# Brief Index: 5698

        
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
        "Controls: Arrow keys to move. Hold Shift to defend. Press Space to attack."
    )

    game_description = (
        "Explore a procedurally generated dungeon, battling enemies and collecting gold to reach the final room."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.UI_HEIGHT = 80
        self.GAME_HEIGHT = self.HEIGHT - self.UI_HEIGHT
        self.TILE_SIZE = 32
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.GAME_HEIGHT // self.TILE_SIZE

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("consolas", 30, bold=True)
            self.font_small = pygame.font.SysFont("consolas", 18)
            self.font_tiny = pygame.font.SysFont("consolas", 14)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 30, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 18)
            self.font_tiny = pygame.font.SysFont("monospace", 14)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_WALL_ACCENT = (80, 80, 100)
        self.COLOR_DOOR = (120, 80, 40)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_ACCENT = (255, 150, 150)
        self.COLOR_ENEMY = (50, 100, 255)
        self.COLOR_ENEMY_ACCENT = (150, 180, 255)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_UI_BG = (40, 40, 55)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH = (50, 205, 50)
        self.COLOR_HEALTH_BG = (139, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        
        # Game constants
        self.MAX_STEPS = 1000
        self.WIN_ROOM = 10
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_ATTACK_DMG = 20
        self.ENEMY_BASE_HEALTH = 50
        self.ENEMY_HEALTH_SCALING = 5
        self.ENEMY_ATTACK_DMG = 10

        # State variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0 # Gold collected
        self.game_over = False
        self.win = False
        self.room_number = 1
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_is_defending = False
        self.door_pos = [0, 0]
        self.enemies = []
        self.gold_piles = []
        self.visual_effects = []
        self.last_action_feedback = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.room_number = 1
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_is_defending = False
        self.visual_effects = []
        self.last_action_feedback = "Welcome, adventurer!"

        self._generate_room()
        
        return self._get_observation(), self._get_info()

    def _generate_room(self):
        self.enemies.clear()
        self.gold_piles.clear()

        # All possible floor tiles
        available_tiles = set(
            (x, y) for x in range(1, self.GRID_WIDTH - 1) for y in range(1, self.GRID_HEIGHT - 1)
        )

        # Place player
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        if tuple(self.player_pos) in available_tiles:
            available_tiles.remove(tuple(self.player_pos))
        
        # Place door (not in a corner)
        side = self.np_random.integers(4)
        if side == 0: # Top
            self.door_pos = [self.np_random.integers(1, self.GRID_WIDTH - 1), 0]
        elif side == 1: # Bottom
            self.door_pos = [self.np_random.integers(1, self.GRID_WIDTH - 1), self.GRID_HEIGHT - 1]
        elif side == 2: # Left
            self.door_pos = [0, self.np_random.integers(1, self.GRID_HEIGHT - 1)]
        else: # Right
            self.door_pos = [self.GRID_WIDTH - 1, self.np_random.integers(1, self.GRID_HEIGHT - 1)]
        if tuple(self.door_pos) in available_tiles:
            available_tiles.remove(tuple(self.door_pos))

        # Place gold
        num_gold = self.np_random.integers(2, 5)
        for _ in range(num_gold):
            if not available_tiles: break
            pos = self.np_random.choice(list(available_tiles))
            self.gold_piles.append(list(pos))
            available_tiles.remove(tuple(pos))

        # Place enemies
        num_enemies = self.np_random.integers(self.room_number, self.room_number + 2)
        enemy_max_health = self.ENEMY_BASE_HEALTH + (self.room_number - 1) * self.ENEMY_HEALTH_SCALING
        for _ in range(num_enemies):
            if not available_tiles: break
            pos = self.np_random.choice(list(available_tiles))
            self.enemies.append({"pos": list(pos), "health": enemy_max_health, "max_health": enemy_max_health})
            available_tiles.remove(tuple(pos))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.visual_effects.clear()
        self.player_is_defending = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player's Turn ---
        action_taken = False
        if space_held: # Attack
            action_taken = True
            # sfx: sword_swing.wav
            self.last_action_feedback = "You attack!"
            self._add_effect('slash', self.player_pos, 1)
            attacked_enemy = False
            for enemy in self.enemies[:]:
                if self._manhattan_distance(self.player_pos, enemy["pos"]) == 1:
                    attacked_enemy = True
                    enemy["health"] -= self.PLAYER_ATTACK_DMG
                    self._add_effect('hit', enemy["pos"], 1)
                    if enemy["health"] <= 0:
                        # sfx: enemy_die.wav
                        reward += 1.0
                        self.enemies.remove(enemy)
                        self.last_action_feedback = "Enemy vanquished!"
            if not attacked_enemy:
                self.last_action_feedback = "You swing at the air."

        elif shift_held: # Defend
            action_taken = True
            # sfx: shield_up.wav
            self.player_is_defending = True
            self.last_action_feedback = "You brace for an attack."

        elif movement != 0: # Move
            action_taken = True
            dist_before = self._manhattan_distance(self.player_pos, self.door_pos)
            
            target_pos = list(self.player_pos)
            if movement == 1: target_pos[1] -= 1 # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right

            if self._is_walkable(target_pos):
                self.player_pos = target_pos
                self.last_action_feedback = "You move."
                # sfx: step.wav
                dist_after = self._manhattan_distance(self.player_pos, self.door_pos)
                if dist_after < dist_before: reward += 0.01
                elif dist_after > dist_before: reward -= 0.01
            else:
                self.last_action_feedback = "You bump into a wall."

        else: # No-op / Wait
            action_taken = True
            self.last_action_feedback = "You wait."
        
        # --- Post-Move Interactions ---
        # Collect gold
        for gold_pos in self.gold_piles[:]:
            if self.player_pos == gold_pos:
                # sfx: coin_collect.wav
                self.score += 5
                reward += 0.5
                self.gold_piles.remove(gold_pos)
                self.last_action_feedback = "You found 5 gold!"

        # Enter door
        if self.player_pos == self.door_pos:
            if self.room_number >= self.WIN_ROOM:
                # sfx: victory.wav
                self.win = True
                self.game_over = True
                self.last_action_feedback = "You escaped the dungeon!"
                reward += 100
            else:
                # sfx: door_open.wav
                self.room_number += 1
                self.last_action_feedback = f"You entered Room {self.room_number}."
                self._generate_room()

        # --- Enemies' Turn ---
        if action_taken and not self.game_over:
            occupied_tiles = {tuple(e["pos"]) for e in self.enemies}
            for enemy in self.enemies:
                if self._manhattan_distance(self.player_pos, enemy["pos"]) == 1:
                    # Attack
                    # sfx: enemy_attack.wav
                    damage = self.ENEMY_ATTACK_DMG
                    if self.player_is_defending:
                        damage *= 0.5
                        self.last_action_feedback = "You block the attack!"
                        # sfx: shield_block.wav
                    else:
                        self.last_action_feedback = "An enemy strikes you!"
                    
                    self.player_health -= damage
                    self._add_effect('hit', self.player_pos, 1)
                    if self.player_health <= 0:
                        self.player_health = 0
                        self.game_over = True
                        self.last_action_feedback = "You have been defeated."
                        # sfx: player_death.wav
                        reward -= 100
                        break
                else:
                    # Move
                    target_pos = list(enemy["pos"])
                    dx = self.player_pos[0] - enemy["pos"][0]
                    dy = self.player_pos[1] - enemy["pos"][1]
                    
                    if abs(dx) > abs(dy):
                        target_pos[0] += np.sign(dx)
                    else:
                        target_pos[1] += np.sign(dy)
                    
                    if self._is_walkable(target_pos) and tuple(target_pos) not in occupied_tiles:
                        occupied_tiles.remove(tuple(enemy["pos"]))
                        enemy["pos"] = target_pos
                        occupied_tiles.add(tuple(enemy["pos"]))

        self.steps += 1
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        if terminated and not (self.win or self.game_over): # Max steps reached
             self.last_action_feedback = "You ran out of time!"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_walkable(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False # Out of bounds
        if x == 0 or x == self.GRID_WIDTH - 1 or y == 0 or y == self.GRID_HEIGHT - 1:
            return pos == self.door_pos # Can only be on border if it's the door
        return True # Is on the floor

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _add_effect(self, type, pos, duration):
        self.visual_effects.append({'type': type, 'pos': list(pos), 'duration': duration})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls and floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if not self._is_walkable([x,y]):
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, rect, 2)
        
        # Draw door
        door_rect = (self.door_pos[0] * self.TILE_SIZE, self.door_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_DOOR, door_rect)

        # Draw gold
        for pos in self.gold_piles:
            center = (int(pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2), int(pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2))
            pygame.draw.circle(self.screen, self.COLOR_GOLD, center, self.TILE_SIZE // 3)

        # Draw enemies
        for enemy in self.enemies:
            rect = (enemy["pos"][0] * self.TILE_SIZE + 4, enemy["pos"][1] * self.TILE_SIZE + 4, self.TILE_SIZE - 8, self.TILE_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_ACCENT, rect, 2, border_radius=4)

        # Draw player
        player_rect = (self.player_pos[0] * self.TILE_SIZE + 2, self.player_pos[1] * self.TILE_SIZE + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_ACCENT, player_rect, 2, border_radius=4)
        if self.player_is_defending:
            center = (int(self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2), int(self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.TILE_SIZE // 2, (100, 100, 255, 100))
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.TILE_SIZE // 2, (200, 200, 255))
        
        # Draw effects
        for effect in self.visual_effects:
            pos_px = (int(effect['pos'][0] * self.TILE_SIZE + self.TILE_SIZE / 2), int(effect['pos'][1] * self.TILE_SIZE + self.TILE_SIZE / 2))
            if effect['type'] == 'slash':
                pygame.draw.circle(self.screen, self.COLOR_WHITE, pos_px, self.TILE_SIZE, 3)
            if effect['type'] == 'hit':
                pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], self.TILE_SIZE // 3, (255, 100, 100, 150))


    def _render_ui(self):
        ui_rect = (0, self.GAME_HEIGHT, self.WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_WALL_ACCENT, (0, self.GAME_HEIGHT), (self.WIDTH, self.GAME_HEIGHT), 2)

        # Health Bar
        hp_text = self.font_small.render("HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(hp_text, (15, self.GAME_HEIGHT + 15))
        health_frac = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_bg_rect = (60, self.GAME_HEIGHT + 15, 200, 20)
        bar_fg_rect = (60, self.GAME_HEIGHT + 15, int(200 * health_frac), 20)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bar_bg_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, bar_fg_rect, border_radius=4)

        # Gold
        gold_text = self.font_large.render(f"GOLD: {self.score}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.WIDTH - 220, self.GAME_HEIGHT + 10))

        # Room
        room_text = self.font_large.render(f"ROOM: {self.room_number}/{self.WIN_ROOM}", True, self.COLOR_UI_TEXT)
        self.screen.blit(room_text, (self.WIDTH - 220, self.GAME_HEIGHT + 40))

        # Action Feedback
        feedback_text = self.font_tiny.render(self.last_action_feedback, True, self.COLOR_UI_TEXT)
        feedback_rect = feedback_text.get_rect(center=(self.WIDTH/2, self.HEIGHT - 20))
        self.screen.blit(feedback_text, feedback_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "room_number": self.room_number,
            "player_pos": self.player_pos,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print("      HUMAN PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Close the window to quit.")
    print("="*30 + "\n")

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Key state handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # Since it's turn-based, we only step on a key press
        # We need a simple way to register one action per key press event
        # A more robust human-play loop would handle this better, but for a
        # simple test, we will step every frame a key is held.
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # In turn-based, we wait for the next input, so we don't need a fast clock
        # but for smooth key registration, a small delay is good.
        clock.tick(10)

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()