
# Generated: 2025-08-28T02:18:46.479448
# Source Brief: brief_04412.md
# Brief Index: 4412

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to select tower type. Press space to place the selected tower on the highlighted grid spot."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along the path."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 19, 23)
    COLOR_PATH = (40, 45, 55)
    COLOR_PATH_BORDER = (60, 65, 75)
    COLOR_BASE = (66, 135, 245)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (219, 51, 51)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GRID = (255, 255, 255, 30)
    COLOR_GRID_SELECT = (255, 255, 0, 150)
    COLOR_TOWER_PLACE_FAIL = (255, 0, 0, 150)

    # Game Params
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 30000 # ~16 minutes at 30fps
    MAX_WAVES = 20
    INITIAL_BASE_HEALTH = 100
    INITIAL_RESOURCES = 150
    RESOURCES_PER_WAVE = 100
    RESOURCES_PER_KILL = 5

    TOWER_SPECS = [
        {"color": (0, 255, 127), "range": 70, "damage": 2.5, "fire_rate": 20, "cost": 50, "projectile_speed": 8, "name": "Cannon"},
        {"color": (255, 215, 0), "range": 100, "damage": 1.5, "fire_rate": 10, "cost": 75, "projectile_speed": 10, "name": "Gatling"},
        {"color": (138, 43, 226), "range": 120, "damage": 10, "fire_rate": 60, "cost": 125, "projectile_speed": 6, "name": "Artillery"},
        {"color": (0, 191, 255), "range": 90, "damage": 0, "fire_rate": 30, "cost": 100, "projectile_speed": 0, "name": "Slower"}, # Special tower
    ]

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
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        self._define_path_and_grid()
        
        self.np_random = None
        self.game_over = False
        self.steps = 0
        self.score = 0
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.selected_tower_type = 0
        self.placement_cursor_index = 0
        self.space_was_pressed = False
        self.wave_in_progress = False
        self.wave_cooldown = 0
        self.placement_feedback_timer = 0
        self.placement_feedback_color = self.COLOR_GRID_SELECT

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check

    def _define_path_and_grid(self):
        self.path = [
            (-20, 100), (80, 100), (120, 140), (120, 260), (180, 320),
            (460, 320), (520, 260), (520, 140), (560, 100), (self.SCREEN_WIDTH + 20, 100)
        ]
        self.path_segments_len = [math.dist(self.path[i], self.path[i+1]) for i in range(len(self.path)-1)]
        self.total_path_len = sum(self.path_segments_len)
        
        self.placement_spots = [
            (80, 40), (160, 40), (240, 40), (320, 40), (400, 40), (480, 40),
            (80, 160), (160, 200), (220, 260), (300, 260), (380, 260), (460, 200), (560, 160),
            (80, 360), (160, 360), (240, 360), (320, 360), (400, 360), (480, 360), (560, 360)
        ]
        self.occupied_spots = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 90 # 3 seconds at 30fps
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.occupied_spots.clear()

        self.selected_tower_type = 0
        self.placement_cursor_index = 0
        self.space_was_pressed = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        self._handle_input(action)

        if not self.game_over:
            damage_dealt_this_step = self._update_projectiles()
            self._update_towers()
            reward += self._update_enemies()

            if damage_dealt_this_step:
                reward += 0.1
            else:
                reward -= 0.01

            self._update_particles()
            reward += self._update_waves()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.base_health <= 0:
                reward = -100.0
            elif self.wave_number > self.MAX_WAVES:
                reward = 100.0
            self.game_over = True
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        
        # Action 0 (Movement): Select tower type
        if 1 <= movement <= 4:
            self.selected_tower_type = movement - 1

        # Action 1 (Space): Place tower
        if space_pressed and not self.space_was_pressed:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.placement_cursor_index < len(self.placement_spots):
                pos = self.placement_spots[self.placement_cursor_index]
                if self.resources >= spec["cost"]:
                    self.towers.append({
                        "pos": pos,
                        "type": self.selected_tower_type,
                        "cooldown": 0,
                        "target": None,
                    })
                    self.resources -= spec["cost"]
                    self.occupied_spots.append(pos)
                    # Find next available spot
                    start_idx = self.placement_cursor_index
                    while True:
                        self.placement_cursor_index = (self.placement_cursor_index + 1) % len(self.placement_spots)
                        if self.placement_spots[self.placement_cursor_index] not in self.occupied_spots:
                            break
                        if self.placement_cursor_index == start_idx: # All spots filled
                            self.placement_cursor_index = len(self.placement_spots) # No more valid spots
                            break
                else: # Not enough resources
                    self.placement_feedback_timer = 15
                    self.placement_feedback_color = self.COLOR_TOWER_PLACE_FAIL

        self.space_was_pressed = bool(space_pressed)

    def _update_waves(self):
        reward = 0
        if not self.wave_in_progress and not self.game_over:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and self.wave_number <= self.MAX_WAVES:
                self.wave_number += 1
                if self.wave_number > 1: # Don't reward for spawning wave 1
                    reward += 5.0 # Wave survival bonus
                    self.resources += self.RESOURCES_PER_WAVE
                if self.wave_number <= self.MAX_WAVES:
                    self._spawn_wave()
                    self.wave_in_progress = True
        elif self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.wave_cooldown = 150 # 5 seconds
        return reward

    def _spawn_wave(self):
        num_enemies = 5 + self.wave_number * 2
        base_health = 10 + self.wave_number * 5
        base_speed = 0.8 + self.wave_number * 0.05
        
        for i in range(num_enemies):
            health = base_health * (1 + self.np_random.uniform(-0.1, 0.1))
            speed = base_speed * (1 + self.np_random.uniform(-0.1, 0.1))
            
            self.enemies.append({
                "pos": self.path[0],
                "health": health,
                "max_health": health,
                "speed": speed,
                "path_idx": 0,
                "dist_on_path": -i * 20, # Stagger enemies
                "slow_timer": 0,
            })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            current_speed = enemy["speed"]
            if enemy["slow_timer"] > 0:
                current_speed *= 0.5 # 50% slow
                enemy["slow_timer"] -= 1

            enemy["dist_on_path"] += current_speed
            
            # Check if enemy reached the end
            if enemy["dist_on_path"] >= self.total_path_len:
                self.base_health -= 10
                self.enemies.remove(enemy)
                # Sound: base_damage.wav
                continue

            # Update position based on distance
            dist = enemy["dist_on_path"]
            temp_dist = 0
            for i in range(len(self.path_segments_len)):
                segment_len = self.path_segments_len[i]
                if temp_dist + segment_len >= dist:
                    progress = (dist - temp_dist) / segment_len
                    start_node = self.path[i]
                    end_node = self.path[i+1]
                    enemy["pos"] = (
                        start_node[0] + (end_node[0] - start_node[0]) * progress,
                        start_node[1] + (end_node[1] - start_node[1]) * progress
                    )
                    enemy["path_idx"] = i
                    break
                temp_dist += segment_len
        return reward

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            spec = self.TOWER_SPECS[tower["type"]]
            
            if tower["cooldown"] == 0:
                # Find best target (furthest along path in range)
                best_target = None
                max_dist = -1
                for enemy in self.enemies:
                    dist_to_enemy = math.dist(tower["pos"], enemy["pos"])
                    if dist_to_enemy <= spec["range"]:
                        if enemy["dist_on_path"] > max_dist:
                            max_dist = enemy["dist_on_path"]
                            best_target = enemy
                
                if best_target:
                    tower["cooldown"] = spec["fire_rate"]
                    if spec["damage"] > 0: # Standard projectile tower
                        self.projectiles.append({
                            "start_pos": tower["pos"],
                            "pos": tower["pos"],
                            "target": best_target,
                            "speed": spec["projectile_speed"],
                            "damage": spec["damage"],
                            "color": spec["color"]
                        })
                        # Sound: shoot.wav
                    elif spec["name"] == "Slower": # Special slow tower
                        best_target["slow_timer"] = max(best_target["slow_timer"], 60) # Apply 2s slow
                        # Create visual effect for slow
                        for _ in range(5):
                            self._create_particle(best_target["pos"], spec["color"], count=1, speed=1, life=20)


    def _update_projectiles(self):
        damage_dealt = False
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj["target"]["pos"]
            direction = (target_pos[0] - proj["pos"][0], target_pos[1] - proj["pos"][1])
            dist = math.hypot(*direction)
            
            if dist < proj["speed"]:
                # Hit
                proj["target"]["health"] -= proj["damage"]
                damage_dealt = True
                self._create_particle(proj["target"]["pos"], proj["color"], count=10, speed=2, life=15)
                # Sound: hit.wav
                
                if proj["target"]["health"] <= 0:
                    self.score += 1.0 # Reward for kill
                    self.resources += self.RESOURCES_PER_KILL
                    self._create_particle(proj["target"]["pos"], self.COLOR_ENEMY, count=20, speed=3, life=25)
                    self.enemies.remove(proj["target"])
                    # Sound: explosion.wav
                
                self.projectiles.remove(proj)
            else:
                # Move projectile
                dx = (direction[0] / dist) * proj["speed"]
                dy = (direction[1] / dist) * proj["speed"]
                proj["pos"] = (proj["pos"][0] + dx, proj["pos"][1] + dy)
        return damage_dealt

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particle(self, pos, color, count, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, 1.0) * speed
            vel = (math.cos(angle) * s, math.sin(angle) * s)
            self.particles.append({"pos": pos, "vel": vel, "life": life, "color": color})
            
    def _check_termination(self):
        return self.base_health <= 0 or self.wave_number > self.MAX_WAVES or self.steps >= self.MAX_STEPS

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
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "enemies": len(self.enemies),
        }

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 24)
        pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, self.path, 26)

        # Draw placement grid and selection
        if self.placement_feedback_timer > 0:
            self.placement_feedback_timer -= 1
            color = self.placement_feedback_color
        else:
            color = self.COLOR_GRID_SELECT
        
        for i, pos in enumerate(self.placement_spots):
            if pos not in self.occupied_spots:
                pygame.gfxdraw.box(self.screen, pygame.Rect(pos[0]-10, pos[1]-10, 20, 20), self.COLOR_GRID)
        
        if self.placement_cursor_index < len(self.placement_spots):
            pos = self.placement_spots[self.placement_cursor_index]
            rect = pygame.Rect(pos[0]-12, pos[1]-12, 24, 24)
            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.gfxdraw.rectangle(self.screen, rect, (*color[:3], 255))

        # Draw base
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DMG
        pygame.draw.rect(self.screen, base_color, (self.SCREEN_WIDTH-20, 80, 20, 40))

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pygame.draw.circle(self.screen, spec["color"], tower["pos"], 10)
            pygame.draw.circle(self.screen, self.COLOR_BG, tower["pos"], 7)
            pygame.draw.circle(self.screen, spec["color"], tower["pos"], 4)
            # Draw range indicator on selected tower
            if self.placement_cursor_index < len(self.placement_spots) and tower["pos"] == self.placement_spots[self.placement_cursor_index]:
                 pygame.gfxdraw.aacircle(self.screen, int(tower["pos"][0]), int(tower["pos"][1]), spec["range"], (*spec["color"], 60))

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 6)
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (255,0,0), (pos[0]-8, pos[1]-12, 16, 3))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0]-8, pos[1]-12, 16 * health_pct, 3))
            if enemy["slow_timer"] > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, (*self.TOWER_SPECS[3]["color"], 150))


        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.circle(self.screen, proj["color"], pos, 3)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, (*proj["color"], 100))

        # Draw particles
        for p in self.particles:
            size = max(1, int(p["life"] / 5))
            pygame.draw.rect(self.screen, p["color"], (p["pos"][0]-size//2, p["pos"][1]-size//2, size, size))

    def _render_ui(self):
        # Top bar
        ui_bar = pygame.Rect(0, 0, self.SCREEN_WIDTH, 30)
        pygame.draw.rect(self.screen, (30,35,45), ui_bar)
        
        info_texts = [
            f"$: {self.resources}",
            f"♥: {self.base_health}",
            f"Wave: {self.wave_number}/{self.MAX_WAVES}",
            f"Score: {int(self.score)}"
        ]
        for i, text in enumerate(info_texts):
            surf = self.font_m.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (10 + i * 150, 5))

        # Tower selection UI
        for i, spec in enumerate(self.TOWER_SPECS):
            x_pos = self.SCREEN_WIDTH - 120
            y_pos = 200 + i * 45
            is_selected = (i == self.selected_tower_type)
            
            box_color = (80,90,110) if is_selected else (50,60,70)
            pygame.draw.rect(self.screen, box_color, (x_pos, y_pos, 110, 40), border_radius=5)
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_TEXT, (x_pos, y_pos, 110, 40), 2, border_radius=5)
            
            pygame.draw.circle(self.screen, spec["color"], (x_pos + 20, y_pos + 20), 8)
            name_surf = self.font_s.render(f"{i+1}:{spec['name']}", True, self.COLOR_TEXT)
            cost_surf = self.font_s.render(f"${spec['cost']}", True, self.COLOR_TEXT)
            self.screen.blit(name_surf, (x_pos + 40, y_pos + 5))
            self.screen.blit(cost_surf, (x_pos + 40, y_pos + 20))

        # Game over / Win screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            text_surf = self.font_l.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(text_surf, text_rect)
            
            final_score_surf = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)
            
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11' or 'dummy' for headless, 'windows' for windows
    
    env = GameEnv()
    env.reset()
    
    # To display the game
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Pygame uses a different coordinate system for surfaces
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate for interactive play
        
        if terminated:
            # Simple reset after a delay
            pygame.time.wait(3000)
            env.reset()
            terminated = False

    env.close()