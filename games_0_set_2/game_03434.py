
# Generated: 2025-08-27T23:21:00.866512
# Source Brief: brief_03434.md
# Brief Index: 3434

        
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
        "Controls: Arrow keys to target quadrants, Space to target center. "
        "Defeat all monsters to win the wave."
    )

    game_description = (
        "Defeat waves of procedurally generated monsters in an isometric arcade arena. "
        "Chain kills to build combos for a higher score."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 63)
        self.COLOR_PLAYER_HEALTH_FG = (46, 204, 113)
        self.COLOR_PLAYER_HEALTH_BG = (60, 60, 60)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_COMBO = (241, 196, 15)
        self.MONSTER_COLORS = [
            (231, 76, 60), (52, 152, 219), (155, 89, 182), (26, 188, 156)
        ]

        # --- Fonts ---
        self.font_main = pygame.font.Font(None, 28)
        self.font_combo = pygame.font.Font(None, 36)
        self.font_floating = pygame.font.Font(None, 22)

        # --- Game Constants ---
        self.ISO_GRID_WIDTH, self.ISO_GRID_HEIGHT = 22, 14
        self.ISO_TILE_WIDTH_HALF, self.ISO_TILE_HEIGHT_HALF = 32, 16
        self.ISO_ORIGIN_X, self.ISO_ORIGIN_Y = self.SCREEN_WIDTH // 2, 80
        
        self.PLAYER_MAX_HEALTH = 100
        self.MONSTER_MAX_HEALTH = 10
        self.MONSTER_COUNT = 15
        self.MONSTER_ATTACK_DAMAGE = 5
        self.MONSTER_ATTACK_INTERVAL = 30
        self.PLAYER_ATTACK_DAMAGE = 10
        self.CLICK_RADIUS = 50
        self.COMBO_WINDOW = 45 # steps
        self.MAX_STEPS = 5000

        self.wave_number = 0
        self.monster_base_speed = 0.04
        
        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, iso_x, iso_y):
        screen_x = self.ISO_ORIGIN_X + (iso_x - iso_y) * self.ISO_TILE_WIDTH_HALF
        screen_y = self.ISO_ORIGIN_Y + (iso_x + iso_y) * self.ISO_TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_MAX_HEALTH

        if options and options.get("wave_clear", False):
            self.wave_number += 1
        else:
            self.wave_number = 1

        self.monster_speed = self.monster_base_speed * (1 + (self.wave_number - 1) * 0.1)

        self.monsters = []
        self.particles = []
        self.floating_texts = []
        self.screen_shake = 0
        self.combo_counter = 0
        self.last_kill_step = -self.COMBO_WINDOW * 2

        self._spawn_monsters()
        return self._get_observation(), self._get_info()

    def _spawn_monsters(self):
        for _ in range(self.MONSTER_COUNT):
            pattern = self.rng.choice(['circle', 'patrol_x', 'patrol_y'])
            start_x = self.rng.uniform(2, self.ISO_GRID_WIDTH - 2)
            start_y = self.rng.uniform(2, self.ISO_GRID_HEIGHT - 2)
            
            monster = {
                "pos": np.array([start_x, start_y], dtype=float),
                "health": self.MONSTER_MAX_HEALTH,
                "alive": True,
                "attack_timer": self.rng.integers(0, self.MONSTER_ATTACK_INTERVAL),
                "pattern": pattern,
                "pattern_phase": self.rng.uniform(0, 2 * math.pi),
                "pattern_radius": self.rng.uniform(2, 4),
                "pattern_center": np.array([start_x, start_y], dtype=float),
                "color": random.choice(self.MONSTER_COLORS),
                "shape_bob": self.rng.uniform(0, 2 * math.pi)
            }
            self.monsters.append(monster)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Small penalty for taking a step

        # 1. Handle player action
        action_reward, damage_dealt = self._handle_action(action)
        reward += action_reward

        # 2. Update monster logic
        self._update_monsters()

        # 3. Update effects
        self._update_effects()

        # 4. Check for termination
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        damage_dealt = 0
        click_pos = None

        quadrant_map = {
            1: (self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.25), # Up -> Top-Left
            4: (self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT * 0.25), # Right -> Top-Right
            3: (self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.75), # Left -> Bottom-Left
            2: (self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT * 0.75), # Down -> Bottom-Right
        }

        if movement in quadrant_map:
            click_pos = quadrant_map[movement]
        elif space_held:
            click_pos = (self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT * 0.5)

        if click_pos:
            # Create click visual effect
            self.particles.append(
                {"pos": list(click_pos), "vel": [0,0], "life": 10, "size": 30, "type": "click"}
            )

            target_monster, min_dist = None, float('inf')
            for m in self.monsters:
                if m["alive"]:
                    m_screen_pos = self._iso_to_screen(m["pos"][0], m["pos"][1])
                    dist = math.hypot(m_screen_pos[0] - click_pos[0], m_screen_pos[1] - click_pos[1])
                    if dist < self.CLICK_RADIUS and dist < min_dist:
                        target_monster, min_dist = m, dist
            
            if target_monster:
                # Hit
                damage = self.PLAYER_ATTACK_DAMAGE
                target_monster["health"] -= damage
                damage_dealt = damage
                reward += 0.1 * damage
                self.score += 10

                # Hit particles
                m_screen_pos = self._iso_to_screen(target_monster["pos"][0], target_monster["pos"][1])
                for _ in range(10):
                    angle = self.rng.uniform(0, 2 * math.pi)
                    speed = self.rng.uniform(1, 4)
                    self.particles.append({
                        "pos": list(m_screen_pos),
                        "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                        "life": self.rng.integers(10, 20), "size": self.rng.uniform(1, 3), "type": "hit"
                    })

                if target_monster["health"] <= 0:
                    # Kill
                    target_monster["alive"] = False
                    reward += 1.0
                    self.score += 100
                    
                    if self.steps - self.last_kill_step < self.COMBO_WINDOW:
                        self.combo_counter += 1
                    else:
                        self.combo_counter = 1
                    
                    combo_reward = 0.5 * self.combo_counter
                    reward += combo_reward
                    self.score += int(50 * self.combo_counter)
                    self.last_kill_step = self.steps
                    self.screen_shake = 10

                    # Floating text for combo
                    if self.combo_counter > 1:
                        self.floating_texts.append({
                            "text": f"x{self.combo_counter} COMBO!", "pos": list(m_screen_pos),
                            "life": 40, "color": self.COLOR_COMBO
                        })
                    
                    # Death particles
                    for _ in range(50):
                        angle = self.rng.uniform(0, 2 * math.pi)
                        speed = self.rng.uniform(2, 8)
                        self.particles.append({
                            "pos": list(m_screen_pos),
                            "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                            "life": self.rng.integers(20, 40), "size": self.rng.uniform(2, 5),
                            "type": "death", "color": target_monster["color"]
                        })
                    # sfx: monster_death.wav
        
        return reward, damage_dealt

    def _update_monsters(self):
        for m in self.monsters:
            if not m["alive"]:
                continue

            # Movement
            m["pattern_phase"] += self.monster_speed
            if m["pattern"] == 'circle':
                m["pos"][0] = m["pattern_center"][0] + math.cos(m["pattern_phase"]) * m["pattern_radius"]
                m["pos"][1] = m["pattern_center"][1] + math.sin(m["pattern_phase"]) * m["pattern_radius"]
            elif m["pattern"] == 'patrol_x':
                m["pos"][0] = m["pattern_center"][0] + math.sin(m["pattern_phase"]) * m["pattern_radius"]
            elif m["pattern"] == 'patrol_y':
                m["pos"][1] = m["pattern_center"][1] + math.sin(m["pattern_phase"]) * m["pattern_radius"]

            m["pos"][0] = np.clip(m["pos"][0], 0, self.ISO_GRID_WIDTH)
            m["pos"][1] = np.clip(m["pos"][1], 0, self.ISO_GRID_HEIGHT)

            # Attack
            m["attack_timer"] += 1
            if m["attack_timer"] >= self.MONSTER_ATTACK_INTERVAL:
                m["attack_timer"] = 0
                self.player_health -= self.MONSTER_ATTACK_DAMAGE
                self.score = max(0, self.score - 25)
                self.screen_shake = 5
                
                # sfx: player_damage.wav
                # Create attack particle
                m_screen_pos = self._iso_to_screen(m["pos"][0], m["pos"][1])
                self.particles.append({
                    "pos": list(m_screen_pos), "vel": [0,0], "life": 15, "size": 0,
                    "type": "enemy_attack", "target_pos": (80, 25)
                })

    def _update_effects(self):
        if self.screen_shake > 0:
            self.screen_shake -= 1

        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if p["type"] != "enemy_attack":
                 p["size"] *= 0.95

        self.floating_texts = [ft for ft in self.floating_texts if ft["life"] > 0]
        for ft in self.floating_texts:
            ft["life"] -= 1
            ft["pos"][1] -= 0.5

    def _check_termination(self):
        if self.player_health <= 0:
            return True, -100.0
        
        if all(not m["alive"] for m in self.monsters):
            return True, 100.0
        
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
            
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = self.rng.integers(-self.screen_shake, self.screen_shake)
            offset_y = self.rng.integers(-self.screen_shake, self.screen_shake)
        
        self._render_game(offset_x, offset_y)
        self._render_ui(offset_x, offset_y)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, ox, oy):
        # Render grid
        for i in range(self.ISO_GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.ISO_GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (start[0] + ox, start[1] + oy), (end[0] + ox, end[1] + oy))
        for i in range(self.ISO_GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.ISO_GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (start[0] + ox, start[1] + oy), (end[0] + ox, end[1] + oy))
        
        # Render monsters (sorted by y-pos for correct layering)
        sorted_monsters = sorted(self.monsters, key=lambda m: m["pos"][1])
        for m in sorted_monsters:
            screen_pos = self._iso_to_screen(m["pos"][0], m["pos"][1])
            x, y = screen_pos[0] + ox, screen_pos[1] + oy
            
            bob = math.sin(m["shape_bob"] + self.steps * 0.1) * 3
            y += int(bob)

            shadow_pos = (x, y + 15)
            pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], 12, 6, (0,0,0,100))
            
            if m["alive"]:
                # Draw monster shape (a triangle)
                points = [
                    (x, y - 12),
                    (x - 10, y + 8),
                    (x + 10, y + 8)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, m["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, m["color"])
                
                # Health bar
                health_pct = m["health"] / self.MONSTER_MAX_HEALTH
                bar_width = 20
                pygame.draw.rect(self.screen, (60,0,0), (x - bar_width/2, y + 15, bar_width, 4))
                pygame.draw.rect(self.screen, (200,0,0), (x - bar_width/2, y + 15, bar_width * health_pct, 4))

            else: # Dead monster
                points = [ (x, y - 2), (x - 8, y + 5), (x + 8, y + 5) ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID)

        # Render particles
        for p in self.particles:
            px, py = int(p["pos"][0]), int(p["pos"][1])
            size = int(p["size"])
            if size < 1: continue

            if p["type"] == "click":
                alpha = int(max(0, 255 * (p["life"] / 10)))
                pygame.gfxdraw.aacircle(self.screen, px+ox, py+oy, size, (46, 204, 113, alpha))
            elif p["type"] == "hit":
                color = self.COLOR_COMBO
                pygame.draw.circle(self.screen, color, (px + ox, py + oy), size)
            elif p["type"] == "death":
                color = p["color"]
                pygame.draw.circle(self.screen, color, (px + ox, py + oy), size)
            elif p["type"] == "enemy_attack":
                alpha = int(max(0, 255 * (p["life"] / 15)))
                target_x, target_y = p["target_pos"]
                progress = (15 - p["life"]) / 15.0
                curr_x = int(px + (target_x - px) * progress)
                curr_y = int(py + (target_y - py) * progress)
                pygame.gfxdraw.filled_circle(self.screen, curr_x+ox, curr_y+oy, 3, (231, 76, 60, alpha))


        # Render floating texts
        for ft in self.floating_texts:
            alpha = int(max(0, 255 * (ft["life"] / 40)))
            text_surf = self.font_combo.render(ft["text"], True, ft["color"])
            text_surf.set_alpha(alpha)
            pos = (ft["pos"][0] - text_surf.get_width() // 2 + ox, ft["pos"][1] - text_surf.get_height() // 2 + oy)
            self.screen.blit(text_surf, pos)

    def _render_ui(self, ox, oy):
        # Player Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        health_color = self.COLOR_PLAYER_HEALTH_FG
        if health_pct < 0.5: health_color = (230, 126, 34)
        if health_pct < 0.25: health_color = (192, 57, 43)
        bar_w, bar_h = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HEALTH_BG, (10 + ox, 10 + oy, bar_w, bar_h))
        pygame.draw.rect(self.screen, health_color, (10 + ox, 10 + oy, bar_w * health_pct, bar_h))
        health_text = self.font_main.render(f"HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15 + ox, 11 + oy))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10 + ox, 10 + oy))
        
        # Wave
        wave_text = self.font_main.render(f"WAVE: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10 + ox, 40 + oy))

        # Combo
        if self.combo_counter > 1 and self.steps - self.last_kill_step < self.COMBO_WINDOW:
            combo_text = self.font_combo.render(f"x{self.combo_counter}", True, self.COLOR_COMBO)
            pos = (self.SCREEN_WIDTH // 2 - combo_text.get_width() // 2 + ox, 10 + oy)
            self.screen.blit(combo_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "wave": self.wave_number,
            "monsters_left": sum(1 for m in self.monsters if m["alive"]),
            "combo": self.combo_counter
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Map pygame keys to gymnasium actions
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }

    # Use a separate screen for display if running interactively
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Arena")
    
    action = [0, 0, 0] # Default no-op action
    
    while running:
        terminated = False
        truncated = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # For this game, actions are discrete events, not continuous holds
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")
                    action = [0, 0, 0] # Reset to no-op after one step
                elif event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    total_reward = 0

        # Draw the observation from the environment to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print("="*20)
            print(f"EPISODE FINISHED. Final Score: {info['score']}")
            print("Press 'R' to play again or close the window.")
            # Wait for reset command
            wait_for_reset = True
            while wait_for_reset and running:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        wait_for_reset = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

    env.close()