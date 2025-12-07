
# Generated: 2025-08-28T00:23:58.657496
# Source Brief: brief_03780.md
# Brief Index: 3780

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Space to attack in your last moved direction. "
        "Doing nothing for a turn passes time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Battle 15 monsters in a grid-based arena. Defeat them all before your health runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    GRID_AREA_SIZE = 400
    CELL_SIZE = GRID_AREA_SIZE // GRID_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_SIZE) // 2
    GRID_OFFSET_Y = 0

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_PLAYER_GLOW = (255, 100, 100, 50)
    COLOR_MONSTER_A = (255, 165, 0) # Orange
    COLOR_MONSTER_B = (180, 80, 255) # Purple
    COLOR_HEALTH_BAR_BG = (70, 20, 20)
    COLOR_HEALTH_BAR_FG = (50, 200, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_ATTACK = (255, 255, 255)
    COLOR_FLASH = (255, 255, 255)

    # Game Parameters
    MAX_STEPS = 1000
    PLAYER_MAX_HEALTH = 100
    PLAYER_ATTACK_DAMAGE = 10
    MONSTER_COUNT = 15
    MONSTER_MAX_HEALTH = 20
    MONSTER_ATTACK_DAMAGE = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)

        self.player = {}
        self.monsters = []
        self.effects = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward_info = ""

        # This ensures all attributes are initialized before validation
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward_info = ""

        player_start_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.player = {
            "pos": player_start_pos,
            "health": self.PLAYER_MAX_HEALTH,
            "facing": 1,  # 1:up, 2:down, 3:left, 4:right
            "flash_timer": 0,
        }

        self.monsters = []
        occupied_positions = {tuple(player_start_pos)}
        for _ in range(self.MONSTER_COUNT):
            while True:
                pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE)]
                if tuple(pos) not in occupied_positions:
                    occupied_positions.add(tuple(pos))
                    break
            
            monster_type = self.np_random.choice(['A', 'B'])
            self.monsters.append({
                "pos": pos,
                "health": self.MONSTER_MAX_HEALTH,
                "type": monster_type,
                "attack_timer": 0, # Used by type B
                "flash_timer": 0,
            })
            
        self.effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update game state ---
        reward = 0.1  # Survival reward
        self.last_reward_info = "+0.1 (survive)"
        
        # 1. Handle player action (Attack > Move > Wait)
        action_taken = self._handle_player_action(movement, space_held)
        
        # 2. Handle monster turn
        monster_damage = self._handle_monster_turn()
        if monster_damage > 0:
            self.player["health"] -= monster_damage
            self.player["flash_timer"] = 3 # Flash for 3 frames

        # 3. Update effects and timers
        self._update_timers_and_effects()

        # 4. Calculate rewards from events
        monsters_killed = self._cull_dead_monsters()
        if monsters_killed > 0:
            kill_reward = monsters_killed * 10
            reward += kill_reward
            self.score += kill_reward
            self.last_reward_info = f"+{kill_reward:.1f} (kill)"

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.player["health"] <= 0:
                loss_penalty = -100
                reward += loss_penalty
                self.score += loss_penalty
                self.last_reward_info = f"{loss_penalty:.1f} (died)"
            elif not self.monsters:
                win_bonus = 100
                reward += win_bonus
                self.score += win_bonus
                self.last_reward_info = f"+{win_bonus:.1f} (win)"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, movement, space_held):
        action_taken = False
        if space_held: # Attack takes priority
            # sfx: player_attack.wav
            action_taken = True
            target_pos = list(self.player["pos"])
            if self.player["facing"] == 1: target_pos[1] -= 1 # Up
            elif self.player["facing"] == 2: target_pos[1] += 1 # Down
            elif self.player["facing"] == 3: target_pos[0] -= 1 # Left
            elif self.player["facing"] == 4: target_pos[0] += 1 # Right

            self.effects.append({"type": "attack", "start": self.player["pos"], "end": target_pos, "life": 2})

            for monster in self.monsters:
                if monster["pos"] == target_pos:
                    monster["health"] -= self.PLAYER_ATTACK_DAMAGE
                    monster["flash_timer"] = 3
                    # sfx: monster_hit.wav
                    break
        
        elif movement > 0: # Move if not attacking
            # sfx: player_move.wav
            action_taken = True
            new_pos = list(self.player["pos"])
            if movement == 1: new_pos[1] -= 1 # Up
            elif movement == 2: new_pos[1] += 1 # Down
            elif movement == 3: new_pos[0] -= 1 # Left
            elif movement == 4: new_pos[0] += 1 # Right
            
            self.player["facing"] = movement

            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player["pos"] = new_pos
        
        # If no attack or move, it's a "wait" turn.
        return action_taken

    def _handle_monster_turn(self):
        total_damage = 0
        player_pos_tuple = tuple(self.player["pos"])
        for monster in self.monsters:
            dist = abs(monster["pos"][0] - player_pos_tuple[0]) + abs(monster["pos"][1] - player_pos_tuple[1])
            if dist == 1: # Is adjacent
                should_attack = False
                if monster["type"] == 'A': # Attacks every turn
                    should_attack = True
                elif monster["type"] == 'B': # Attacks every 2 turns
                    if monster["attack_timer"] <= 0:
                        should_attack = True
                        monster["attack_timer"] = 2 # Reset timer
                    
                if should_attack:
                    total_damage += self.MONSTER_ATTACK_DAMAGE
                    # sfx: player_hit.wav
                    self.effects.append({"type": "attack", "start": monster["pos"], "end": self.player["pos"], "life": 2})

            if monster["type"] == 'B' and monster["attack_timer"] > 0:
                monster["attack_timer"] -= 1

        return total_damage

    def _cull_dead_monsters(self):
        killed_count = 0
        alive_monsters = []
        for m in self.monsters:
            if m["health"] > 0:
                alive_monsters.append(m)
            else:
                # sfx: monster_die.wav
                for _ in range(10): # Death particle effect
                    self.effects.append({
                        "type": "particle", "pos": m["pos"], "life": self.np_random.integers(5, 15),
                        "vel": [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)],
                        "color": self.COLOR_MONSTER_A if m["type"] == 'A' else self.COLOR_MONSTER_B
                    })
                killed_count += 1
        self.monsters = alive_monsters
        return killed_count

    def _update_timers_and_effects(self):
        # Player flash
        if self.player["flash_timer"] > 0:
            self.player["flash_timer"] -= 1
        
        # Monster flash
        for m in self.monsters:
            if m["flash_timer"] > 0:
                m["flash_timer"] -= 1

        # Visual effects
        active_effects = []
        for effect in self.effects:
            effect["life"] -= 1
            if effect["type"] == "particle":
                effect["pos"][0] += effect["vel"][0] * 0.1
                effect["pos"][1] += effect["vel"][1] * 0.1
            if effect["life"] > 0:
                active_effects.append(effect)
        self.effects = active_effects

    def _check_termination(self):
        if self.player["health"] <= 0:
            self.game_over = True
            return True
        if not self.monsters: # No monsters left
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "player_health": self.player["health"],
            "monsters_left": len(self.monsters),
        }

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            start_x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            start_y = self.GRID_OFFSET_Y
            end_x = start_x
            end_y = self.GRID_OFFSET_Y + self.GRID_AREA_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))
        for i in range(self.GRID_SIZE + 1):
            start_x = self.GRID_OFFSET_X
            start_y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            end_x = self.GRID_OFFSET_X + self.GRID_AREA_SIZE
            end_y = start_y
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y))

        # Draw monsters
        for monster in self.monsters:
            px, py = self._grid_to_pixel(monster["pos"])
            color = self.COLOR_MONSTER_A if monster["type"] == 'A' else self.COLOR_MONSTER_B
            
            # Health bar
            bar_w = self.CELL_SIZE * 0.8
            bar_h = 5
            health_pct = max(0, monster["health"] / self.MONSTER_MAX_HEALTH)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (px - bar_w/2, py - self.CELL_SIZE/2 - bar_h - 2, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (px - bar_w/2, py - self.CELL_SIZE/2 - bar_h - 2, bar_w * health_pct, bar_h))
            
            # Body
            size = self.CELL_SIZE * 0.8
            body_rect = pygame.Rect(px - size/2, py - size/2, size, size)
            pygame.draw.rect(self.screen, color, body_rect, border_radius=4)
            if monster["flash_timer"] > 0:
                pygame.draw.rect(self.screen, self.COLOR_FLASH, body_rect, border_radius=4)
            
            # Type B charge indicator
            if monster["type"] == 'B' and monster["attack_timer"] == 1:
                pygame.gfxdraw.filled_circle(self.screen, px, py, int(size*0.2), (255,255,255,150))


        # Draw player
        px, py = self._grid_to_pixel(self.player["pos"])
        size = self.CELL_SIZE * 0.9 # Slightly larger
        
        # Glow
        glow_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (size, size), size)
        self.screen.blit(glow_surf, (px - size, py - size))
        
        # Body
        body_rect = pygame.Rect(px - size/2, py - size/2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=6)
        if self.player["flash_timer"] > 0:
             pygame.draw.rect(self.screen, self.COLOR_FLASH, body_rect, border_radius=6)

        # Facing indicator
        facing_pos = [px, py]
        if self.player["facing"] == 1: facing_pos[1] -= size * 0.3
        elif self.player["facing"] == 2: facing_pos[1] += size * 0.3
        elif self.player["facing"] == 3: facing_pos[0] -= size * 0.3
        elif self.player["facing"] == 4: facing_pos[0] += size * 0.3
        pygame.draw.circle(self.screen, (255, 255, 255), (int(facing_pos[0]), int(facing_pos[1])), 3)
        
        # Draw effects
        for effect in self.effects:
            if effect["type"] == "attack":
                start_px = self._grid_to_pixel(effect["start"])
                end_px = self._grid_to_pixel(effect["end"])
                alpha = 255 * (effect["life"] / 2.0)
                pygame.draw.line(self.screen, (*self.COLOR_ATTACK, alpha), start_px, end_px, 4)
            elif effect["type"] == "particle":
                p_px, p_py = self._grid_to_pixel(effect["pos"])
                size = max(1, int(5 * (effect["life"] / 15.0)))
                pygame.draw.circle(self.screen, effect["color"], (p_px, p_py), size)

    def _render_ui(self):
        # Player Health Bar
        health_bar_width = 200
        health_bar_height = 25
        health_x = 20
        health_y = self.SCREEN_HEIGHT - health_bar_height - 15
        
        health_pct = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (health_x, health_y, health_bar_width, health_bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (health_x, health_y, health_bar_width * health_pct, health_bar_height), border_radius=5)
        
        health_text = self.font_large.render(f"HP: {self.player['health']}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (health_x + 10, health_y))

        # Monster Count
        monster_text = self.font_ui.render(f"Monsters: {len(self.monsters)}/{self.MONSTER_COUNT}", True, self.COLOR_TEXT)
        text_rect = monster_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(monster_text, text_rect)
        
        # Step Count
        step_text = self.font_ui.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        text_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 35))
        self.screen.blit(step_text, text_rect)

        # Last Reward Info
        reward_text = self.font_small.render(f"Reward: {self.last_reward_info}", True, self.COLOR_TEXT)
        text_rect = reward_text.get_rect(bottomleft=(20, self.SCREEN_HEIGHT - 50))
        self.screen.blit(reward_text, text_rect)

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Grid Arena Battle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement = 0  # 0=none
        space = 0     # 0=released
        shift = 0     # 0=released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        # The game is turn-based, so we only step on a key press
        action_key_pressed = any([
            keys[pygame.K_UP], keys[pygame.K_DOWN], keys[pygame.K_LEFT], 
            keys[pygame.K_RIGHT], keys[pygame.K_SPACE]
        ])
        
        # Allow a "wait" action with the 'w' key
        if keys[pygame.K_w]:
             action_key_pressed = True
             movement, space, shift = 0, 0, 0

        if action_key_pressed:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # In manual play, we need a delay to make it playable
        pygame.time.wait(100)
        
        if terminated:
            print("Game Over!")
            pygame.time.wait(2000) # Show final screen for 2 seconds

    pygame.quit()