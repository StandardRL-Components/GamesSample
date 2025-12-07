
# Generated: 2025-08-28T02:39:41.557901
# Source Brief: brief_04524.md
# Brief Index: 4524

        
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
        "Controls: Arrow keys to move. Hold Space and press an arrow key to squash in that direction. Hold Shift to dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Squash all the monsters in a top-down arcade arena before they get you! You have 3 lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.NUM_MONSTERS = 15
        self.PLAYER_INITIAL_HEALTH = 3

        # Colors
        self.COLOR_BG = (32, 32, 32)
        self.COLOR_WALL = (220, 220, 220)
        self.COLOR_PLAYER = (50, 205, 50)
        self.COLOR_PLAYER_INNER = (144, 238, 144)
        self.COLOR_MONSTER_TYPES = [(255, 70, 70), (70, 150, 255), (255, 255, 80)] # Red, Blue, Yellow
        self.COLOR_ATTACK = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        
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
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_game_over = pygame.font.SysFont(None, 75)

        # Initialize state variables
        self.player_pos = None
        self.player_health = 0
        self.player_last_move_dir = (0, 0)
        self.player_invulnerable_timer = 0
        self.attack_cooldown = 0
        self.dash_cooldown = 0
        self.monsters = []
        self.particles = []
        self.attack_visual = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.player_health = self.PLAYER_INITIAL_HEALTH
        self.player_last_move_dir = (0, -1) # Default up
        self.player_invulnerable_timer = 60 # Brief invulnerability at start
        self.attack_cooldown = 0
        self.dash_cooldown = 0

        # Game elements
        self.particles = []
        self.attack_visual = None
        
        # Monster state
        self.monsters = []
        occupied_positions = {self.player_pos}
        for _ in range(self.NUM_MONSTERS):
            while True:
                pos = (self.np_random.integers(1, self.GRID_W - 1), self.np_random.integers(1, self.GRID_H - 1))
                if pos not in occupied_positions:
                    occupied_positions.add(pos)
                    break
            
            monster_type_idx = self.np_random.integers(0, 3)
            monster = {
                "pos": pos,
                "alive": True,
                "color": self.COLOR_MONSTER_TYPES[monster_type_idx],
                "ai_type": ["pacer", "random", "chaser"][monster_type_idx],
                "ai_state": 0 if monster_type_idx == 0 else self.np_random.integers(0, 4)
            }
            self.monsters.append(monster)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty for existing
        
        self._update_cooldowns_and_effects()

        # Unpack factorized action
        movement_action = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Handle player actions
        reward += self._handle_input(movement_action, space_held, shift_held)

        # Update game world
        if self.steps % 4 == 0: # Monsters move slower than player
             self._update_monsters()
        
        # Check for collisions and game events
        reward += self._check_events()

        # Update step counter and check for termination
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100 # Penalty for dying
            elif sum(m["alive"] for m in self.monsters) == 0:
                reward += 100 # Bonus for winning
            else: # Timed out
                reward -= 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement_action, space_held, shift_held):
        reward = 0
        
        # 1. Attack action
        if space_held and self.attack_cooldown == 0:
            attack_dir = self.player_last_move_dir
            if movement_action != 0:
                move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
                attack_dir = move_map[movement_action]
            
            attack_pos = (self.player_pos[0] + attack_dir[0], self.player_pos[1] + attack_dir[1])
            self.attack_visual = {"pos": attack_pos, "timer": 5}
            self.attack_cooldown = 15 # 0.5s cooldown
            # sfx: player_attack.wav
            
            squashed_monster = False
            for monster in self.monsters:
                if monster["alive"] and monster["pos"] == attack_pos:
                    monster["alive"] = False
                    self.score += 10
                    reward += 10
                    self._create_particles(monster["pos"], monster["color"])
                    squashed_monster = True
                    # sfx: monster_squash.wav
                    break # Only squash one monster per attack

        # 2. Movement action
        if movement_action != 0:
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            move_dir = move_map[movement_action]
            self.player_last_move_dir = move_dir
            
            dash_multiplier = 1
            if shift_held and self.dash_cooldown == 0:
                dash_multiplier = 2
                self.dash_cooldown = 30 # 1s cooldown
            
            new_pos_x = self.player_pos[0] + move_dir[0] * dash_multiplier
            new_pos_y = self.player_pos[1] + move_dir[1] * dash_multiplier
            
            # Check wall collision
            if 1 <= new_pos_x < self.GRID_W - 1 and 1 <= new_pos_y < self.GRID_H - 1:
                self.player_pos = (new_pos_x, new_pos_y)
            else:
                reward -= 0.1 # Penalty for hitting a wall

        return reward

    def _update_monsters(self):
        for monster in self.monsters:
            if not monster["alive"]:
                continue

            pos = monster["pos"]
            
            if monster["ai_type"] == "pacer":
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if monster["ai_state"] >= 4: monster["ai_state"] = 0
                move_dir = dirs[monster["ai_state"]]
                new_pos = (pos[0] + move_dir[0], pos[1] + move_dir[1])
                if 1 <= new_pos[0] < self.GRID_W - 1 and 1 <= new_pos[1] < self.GRID_H - 1:
                    monster["pos"] = new_pos
                else: # Hit a wall, reverse direction
                    monster["ai_state"] = (monster["ai_state"] + 1) % 4
            
            elif monster["ai_type"] == "random":
                if self.np_random.random() < 0.5: # 50% chance to change direction
                    monster["ai_state"] = self.np_random.integers(0, 4)
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                move_dir = dirs[monster["ai_state"]]
                new_pos = (pos[0] + move_dir[0], pos[1] + move_dir[1])
                if 1 <= new_pos[0] < self.GRID_W - 1 and 1 <= new_pos[1] < self.GRID_H - 1:
                    monster["pos"] = new_pos
                else: # Hit a wall, pick a new random direction
                    monster["ai_state"] = self.np_random.integers(0, 4)

            elif monster["ai_type"] == "chaser":
                dx, dy = self.player_pos[0] - pos[0], self.player_pos[1] - pos[1]
                dist = abs(dx) + abs(dy)
                if 1 < dist < 8: # Chase within a certain range
                    if abs(dx) > abs(dy):
                        new_pos = (pos[0] + np.sign(dx), pos[1])
                    else:
                        new_pos = (pos[0], pos[1] + np.sign(dy))
                    monster["pos"] = new_pos

    def _check_events(self):
        reward = 0
        if self.player_invulnerable_timer == 0:
            for monster in self.monsters:
                if monster["alive"] and monster["pos"] == self.player_pos:
                    self.player_health -= 1
                    self.player_invulnerable_timer = 45 # 1.5s invulnerability
                    reward -= 20
                    self.score = max(0, self.score - 5)
                    # sfx: player_hit.wav
                    break # Only take damage from one monster at a time
        return reward

    def _update_cooldowns_and_effects(self):
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.dash_cooldown > 0: self.dash_cooldown -= 1
        if self.player_invulnerable_timer > 0: self.player_invulnerable_timer -= 1
        if self.attack_visual and self.attack_visual["timer"] > 0:
            self.attack_visual["timer"] -= 1
        else:
            self.attack_visual = None
        
        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["life"] -= 1

    def _check_termination(self):
        if self.player_health <= 0:
            return True
        if all(not m["alive"] for m in self.monsters):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _create_particles(self, grid_pos, color):
        px_pos = ((grid_pos[0] + 0.5) * self.GRID_SIZE, (grid_pos[1] + 0.5) * self.GRID_SIZE)
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(10, 20)
            self.particles.append({"pos": px_pos, "vel": vel, "life": life, "color": color})
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        wall_rects = [
            pygame.Rect(0, 0, self.WIDTH, self.GRID_SIZE),
            pygame.Rect(0, 0, self.GRID_SIZE, self.HEIGHT),
            pygame.Rect(self.WIDTH - self.GRID_SIZE, 0, self.GRID_SIZE, self.HEIGHT),
            pygame.Rect(0, self.HEIGHT - self.GRID_SIZE, self.WIDTH, self.GRID_SIZE)
        ]
        for rect in wall_rects:
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw monsters
        for monster in self.monsters:
            if monster["alive"]:
                m_rect = pygame.Rect(monster["pos"][0] * self.GRID_SIZE, monster["pos"][1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, monster["color"], m_rect)

        # Draw player
        is_invulnerable_flash = self.player_invulnerable_timer > 0 and (self.steps // 3) % 2 == 0
        if not is_invulnerable_flash:
            p_rect = pygame.Rect(self.player_pos[0] * self.GRID_SIZE, self.player_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_INNER, p_rect.inflate(-6, -6))

        # Draw attack visual
        if self.attack_visual:
            alpha = int(255 * (self.attack_visual["timer"] / 5))
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, (self.attack_visual["pos"][0] * self.GRID_SIZE, self.attack_visual["pos"][1] * self.GRID_SIZE))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = p["color"] + (alpha,)
            size = max(1, int(self.GRID_SIZE / 4 * (p["life"] / 20)))
            pygame.gfxdraw.box(self.screen, (int(p["pos"][0] - size/2), int(p["pos"][1] - size/2), size, size), color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", False, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.GRID_SIZE + 10, 2))

        # Health
        health_text = self.font_ui.render(f"LIVES: {self.player_health}", False, self.COLOR_UI_TEXT)
        health_rect = health_text.get_rect(topright=(self.WIDTH - self.GRID_SIZE - 10, 2))
        self.screen.blit(health_text, health_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = sum(m["alive"] for m in self.monsters) == 0
            msg = "YOU WIN!" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "monsters_left": sum(m["alive"] for m in self.monsters),
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Monster Squasher")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0
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
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Wait 3 seconds before closing

    env.close()