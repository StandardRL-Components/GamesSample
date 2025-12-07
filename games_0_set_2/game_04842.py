
# Generated: 2025-08-28T03:10:33.828638
# Source Brief: brief_04842.md
# Brief Index: 4842

        
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
        "Controls: Arrow keys to move. Hold space to attack the nearest enemy. Hold shift to attack the weakest enemy."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot in a grid-based arena, strategically battling 15 enemies to achieve ultimate victory."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_LASER = (255, 200, 0)
    COLOR_PARTICLE = (255, 255, 150)
    COLOR_WHITE = (255, 255, 255)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40

    # Game Parameters
    MAX_STEPS = 1000
    NUM_ENEMIES = 15
    PLAYER_MAX_HEALTH = 100
    ENEMY_MAX_HEALTH = 20
    PLAYER_ATTACK_DAMAGE = 10
    
    # Cooldowns (in steps/frames)
    PLAYER_MOVE_COOLDOWN_MAX = 8
    PLAYER_ATTACK_COOLDOWN_MAX = 15
    ENEMY_MOVE_COOLDOWN_MAX = 20

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        
        self.player = None
        self.enemies = []
        self.effects = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.effects = []

        # Player setup
        start_pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
        self.player = {
            "grid_pos": np.array(start_pos),
            "pixel_pos": np.array(start_pos, dtype=float) * self.CELL_SIZE + self.CELL_SIZE / 2,
            "health": self.PLAYER_MAX_HEALTH,
            "move_cooldown": 0,
            "attack_cooldown": 0,
            "hit_timer": 0,
        }

        # Enemy setup
        self.enemies = []
        occupied_positions = {tuple(self.player["grid_pos"])}
        for _ in range(self.NUM_ENEMIES):
            while True:
                pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
                if pos not in occupied_positions:
                    occupied_positions.add(pos)
                    break
            
            self.enemies.append({
                "grid_pos": np.array(pos),
                "pixel_pos": np.array(pos, dtype=float) * self.CELL_SIZE + self.CELL_SIZE / 2,
                "health": self.ENEMY_MAX_HEALTH,
                "move_cooldown": self.np_random.integers(0, self.ENEMY_MOVE_COOLDOWN_MAX),
                "hit_timer": 0,
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for time passing to encourage efficiency
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # --- Update Cooldowns ---
        self.player["move_cooldown"] = max(0, self.player["move_cooldown"] - 1)
        self.player["attack_cooldown"] = max(0, self.player["attack_cooldown"] - 1)
        for enemy in self.enemies:
            enemy["move_cooldown"] = max(0, enemy["move_cooldown"] - 1)
            enemy["hit_timer"] = max(0, enemy["hit_timer"] - 1)
        self.player["hit_timer"] = max(0, self.player["hit_timer"] - 1)

        # --- Handle Player Input ---
        reward += self._handle_player_action(action)

        # --- Update Enemies ---
        self._update_enemies()
        
        # --- Update Game Logic ---
        self._update_positions()
        reward += self._process_attacks_and_defeats()
        self._update_effects()

        # --- Check Termination ---
        terminated = False
        if len(self.enemies) == 0:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Movement ---
        if movement != 0 and self.player["move_cooldown"] == 0:
            move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement)
            if move_dir:
                old_pos = self.player["grid_pos"]
                new_pos = old_pos + np.array(move_dir)
                if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                    self.player["grid_pos"] = new_pos
                    self.player["move_cooldown"] = self.PLAYER_MOVE_COOLDOWN_MAX
                    
                    # Reward for risky/safe moves
                    nearest_enemy = self._get_nearest_enemy(self.player["grid_pos"])
                    if nearest_enemy:
                        dist_before = np.linalg.norm(old_pos - nearest_enemy["grid_pos"])
                        dist_after = np.linalg.norm(new_pos - nearest_enemy["grid_pos"])
                        if dist_after < dist_before:
                            reward += 0.2  # Risky move
                        else:
                            reward -= 0.02 # Safe move

        # --- Attack ---
        if self.player["attack_cooldown"] == 0 and (space_held or shift_held):
            # Find enemies in 3x3 range
            in_range_enemies = [
                e for e in self.enemies 
                if np.max(np.abs(self.player["grid_pos"] - e["grid_pos"])) <= 1
            ]
            
            target = None
            if shift_held: # Prioritize shift: attack lowest health
                if in_range_enemies:
                    target = min(in_range_enemies, key=lambda e: e["health"])
            elif space_held: # Attack nearest
                if in_range_enemies:
                    target = min(in_range_enemies, key=lambda e: np.linalg.norm(self.player["grid_pos"] - e["grid_pos"]))
            
            if target:
                # SFX: LaserFire.wav
                target["health"] -= self.PLAYER_ATTACK_DAMAGE
                target["hit_timer"] = 5 # Flash for 5 frames
                self.player["attack_cooldown"] = self.PLAYER_ATTACK_COOLDOWN_MAX
                
                # Add laser effect
                self.effects.append({
                    "type": "laser",
                    "start": self.player["pixel_pos"],
                    "end": target["pixel_pos"],
                    "life": 4,
                })

        return reward

    def _update_enemies(self):
        occupied = {tuple(e["grid_pos"]) for e in self.enemies}
        occupied.add(tuple(self.player["grid_pos"]))

        for enemy in self.enemies:
            if enemy["move_cooldown"] == 0:
                enemy["move_cooldown"] = self.ENEMY_MOVE_COOLDOWN_MAX + self.np_random.integers(-5, 5)
                
                possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                self.np_random.shuffle(possible_moves)

                for move in possible_moves:
                    new_pos = enemy["grid_pos"] + np.array(move)
                    if (0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H and tuple(new_pos) not in occupied):
                        occupied.remove(tuple(enemy["grid_pos"]))
                        enemy["grid_pos"] = new_pos
                        occupied.add(tuple(new_pos))
                        break

    def _update_positions(self):
        # Interpolate player position
        target_pixel_pos = self.player["grid_pos"] * self.CELL_SIZE + self.CELL_SIZE / 2
        self.player["pixel_pos"] += (target_pixel_pos - self.player["pixel_pos"]) * 0.4

        # Interpolate enemy positions
        for enemy in self.enemies:
            target_pixel_pos = enemy["grid_pos"] * self.CELL_SIZE + self.CELL_SIZE / 2
            enemy["pixel_pos"] += (target_pixel_pos - enemy["pixel_pos"]) * 0.4

    def _process_attacks_and_defeats(self):
        reward = 0
        defeated_enemies = [e for e in self.enemies if e["health"] <= 0]
        
        for enemy in defeated_enemies:
            # SFX: Explosion.wav
            reward += 10
            self.score += 100
            
            # Add explosion particles
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                self.effects.append({
                    "type": "particle",
                    "pos": enemy["pixel_pos"].copy(),
                    "vel": vel,
                    "life": self.np_random.integers(10, 20),
                    "size": self.np_random.uniform(1, 4)
                })

        self.enemies = [e for e in self.enemies if e["health"] > 0]
        return reward

    def _update_effects(self):
        for effect in self.effects:
            effect["life"] -= 1
            if effect["type"] == "particle":
                effect["pos"] += effect["vel"]
        self.effects = [e for e in self.effects if e["life"] > 0]

    def _get_nearest_enemy(self, pos):
        if not self.enemies:
            return None
        return min(self.enemies, key=lambda e: np.linalg.norm(pos - e["grid_pos"]))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw enemies
        for enemy in self.enemies:
            pos = enemy["pixel_pos"].astype(int)
            size = int(self.CELL_SIZE * 0.35)
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 4, self.COLOR_ENEMY_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
            if enemy["hit_timer"] > 0:
                 pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, (255, 255, 255, 150))


        # Draw player
        pos = self.player["pixel_pos"].astype(int)
        size = int(self.CELL_SIZE * 0.4)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 6, self.COLOR_PLAYER_GLOW)
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (pos[0] - size, pos[1] - size, size*2, size*2))
        if self.player["hit_timer"] > 0:
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (pos[0] - size, pos[1] - size, size*2, size*2), 3)

        # Draw effects
        for effect in self.effects:
            if effect["type"] == "laser":
                start = effect["start"].astype(int)
                end = effect["end"].astype(int)
                alpha = int(255 * (effect["life"] / 4.0))
                color = (*self.COLOR_LASER, alpha)
                pygame.draw.line(self.screen, color, start, end, max(1, effect["life"]))
            elif effect["type"] == "particle":
                pos = effect["pos"].astype(int)
                alpha = int(255 * (effect["life"] / 20.0))
                color = (*self.COLOR_PARTICLE, alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(effect["size"]), color)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        health_bar_width = 200
        health_bar_rect = pygame.Rect(15, 15, health_bar_width, 20)
        fill_width = int(health_bar_width * health_pct)
        fill_rect = pygame.Rect(15, 15, fill_width, 20)
        
        # Interpolate color from green to red
        health_color = (
            self.COLOR_HEALTH_RED[0] + (self.COLOR_HEALTH_GREEN[0] - self.COLOR_HEALTH_RED[0]) * health_pct,
            self.COLOR_HEALTH_GREEN[1] * health_pct,
            0
        )
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_rect)
        pygame.draw.rect(self.screen, health_color, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, health_bar_rect, 1)

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 40))

        # Enemy Count
        enemy_text = self.font_small.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_UI_TEXT)
        text_rect = enemy_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(enemy_text, text_rect)
        
        # Game Over / Win Text
        if self.game_over:
            if len(self.enemies) == 0:
                end_text_str = "VICTORY"
                color = self.COLOR_PLAYER
            else:
                end_text_str = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_large.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_BG, 200))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "enemies_remaining": len(self.enemies),
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Robot Combat")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to restart.")
            
        clock.tick(30) # Run at 30 FPS

    env.close()