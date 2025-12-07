import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by placing defensive towers along their path."
    )

    # Frames auto-advance for smooth real-time waves.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000  # Increased from brief's 1000 to allow for a full game

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_BASE = (0, 255, 128)
    COLOR_BASE_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 40, 55, 180)
    COLOR_CURSOR_VALID = (0, 255, 128, 100)
    COLOR_CURSOR_INVALID = (255, 50, 50, 100)

    TOWER_SPECS = {
        0: {"name": "Gun Turret", "cost": 100, "range": 80, "damage": 2.5, "fire_rate": 0.3, "color": (0, 150, 255), "proj_color": (100, 200, 255)},
        1: {"name": "Cannon", "cost": 250, "range": 120, "damage": 15, "fire_rate": 1.5, "color": (255, 150, 0), "proj_color": (255, 200, 100)},
    }

    # Game Phases
    PHASE_INTERMISSION = 0
    PHASE_WAVE_ACTIVE = 1

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

        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.np_random = None

        # This validation function is called on instantiation.
        # It needs a reset environment to work correctly.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.base_pos = (self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT // 2)
        self.base_health = 100
        self.max_base_health = 100

        self.resources = 250
        self.current_wave = 0
        self.total_waves = 20
        self.game_phase = self.PHASE_INTERMISSION

        self.path = self._generate_path()
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.selected_tower_type = 0

        self.wave_spawn_timer = 0
        self.enemies_to_spawn = []

        self.start_wave_button = pygame.Rect(self.SCREEN_WIDTH - 160, self.SCREEN_HEIGHT - 50, 150, 40)

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1

        self._handle_input(action)
        self._update_game_state()

        terminated = self._check_termination()
        reward = self.reward_this_step
        self.score += reward

        # The truncated value is always False as per the brief.
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # Cursor movement
        cursor_speed = 5
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        if movement == 2: self.cursor_pos[1] += cursor_speed
        if movement == 3: self.cursor_pos[0] -= cursor_speed
        if movement == 4: self.cursor_pos[0] += cursor_speed

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        if self.game_phase == self.PHASE_INTERMISSION:
            # Cycle tower type
            if shift_pressed:
                self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
                # SFX: UI_Cycle

            if space_pressed:
                # Check for starting the wave
                if self.start_wave_button.collidepoint(self.cursor_pos):
                    self._start_next_wave()
                    # SFX: UI_Confirm
                else:  # Try to place a tower
                    self._place_tower()

    def _update_game_state(self):
        if self.game_phase == self.PHASE_WAVE_ACTIVE:
            self._spawn_enemies()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()

            if not self.enemies and not self.enemies_to_spawn:
                self._end_wave()

        self._update_particles()

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources >= spec["cost"] and self._is_valid_placement(self.cursor_pos):
            self.resources -= spec["cost"]
            self.towers.append({
                "pos": tuple(self.cursor_pos),
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None
            })
            # SFX: Tower_Place

    def _is_valid_placement(self, pos):
        # Cannot place on path
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            dist_sq = self._dist_to_segment_sq(pos, p1, p2)
            if dist_sq < (20 * 20):  # 20 pixel buffer around path
                return False
        # Cannot place on other towers
        for tower in self.towers:
            if math.hypot(pos[0] - tower["pos"][0], pos[1] - tower["pos"][1]) < 30:
                return False
        # Cannot place on UI elements
        if self.start_wave_button.collidepoint(pos):
            return False
        return True

    def _start_next_wave(self):
        if self.game_phase == self.PHASE_INTERMISSION:
            self.current_wave += 1
            self.game_phase = self.PHASE_WAVE_ACTIVE

            # Difficulty scaling
            num_enemies = 3 + self.current_wave * 2
            base_health = 10 + (self.current_wave - 1) * 5 * 1.05
            base_speed = 1.0 + (self.current_wave - 1) * 0.05 * 1.02

            for i in range(num_enemies):
                self.enemies_to_spawn.append({
                    "health": base_health,
                    "max_health": base_health,
                    "speed": base_speed,
                    "spawn_delay": i * 15  # Spawn 2 enemies per second
                })

    def _end_wave(self):
        self.game_phase = self.PHASE_INTERMISSION
        wave_bonus = 25 + self.current_wave * 5
        self.resources += wave_bonus
        self.reward_this_step += 1.0
        # SFX: Wave_Complete
        if self.current_wave >= self.total_waves:
            self.game_over = True  # Win condition

    def _spawn_enemies(self):
        if not self.enemies_to_spawn: return

        self.wave_spawn_timer += 1

        if self.wave_spawn_timer >= self.enemies_to_spawn[0]["spawn_delay"]:
            enemy_spec = self.enemies_to_spawn.pop(0)
            self.enemies.append({
                "pos": list(self.path[0]),
                "path_index": 1,
                "health": enemy_spec["health"],
                "max_health": enemy_spec["max_health"],
                "speed": enemy_spec["speed"],
            })
            self.wave_spawn_timer = 0
            # SFX: Enemy_Spawn

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy["path_index"] >= len(self.path):
                self.enemies.remove(enemy)
                damage = 10
                self.base_health -= damage
                self.reward_this_step -= 0.01 * damage
                # SFX: Base_Damage
                self._create_explosion(self.base_pos, self.COLOR_BASE)
                continue

            target_pos = self.path[enemy["path_index"]]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["path_index"] += 1
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1 / self.FPS
                continue

            # Find a target
            target = None
            best_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(tower["pos"][0] - enemy["pos"][0], tower["pos"][1] - enemy["pos"][1])
                if dist <= spec["range"]:
                    # Target enemy furthest along the path
                    enemy_path_dist = enemy["path_index"] + (math.hypot(enemy["pos"][0] - self.path[enemy["path_index"]-1][0], enemy["pos"][1] - self.path[enemy["path_index"]-1][1]))
                    if enemy_path_dist < best_dist:
                        best_dist = enemy_path_dist
                        target = enemy

            if target:
                tower["cooldown"] = spec["fire_rate"]
                self.projectiles.append({
                    "start_pos": tower["pos"],
                    "target": target,
                    "pos": list(tower["pos"]),
                    "speed": 15,
                    "damage": spec["damage"],
                    "color": spec["proj_color"]
                })
                # SFX: Tower_Fire

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < proj["speed"]:
                proj["target"]["health"] -= proj["damage"]
                self.projectiles.remove(proj)
                # SFX: Enemy_Hit
                if proj["target"]["health"] <= 0:
                    self._create_explosion(proj["target"]["pos"], self.COLOR_ENEMY)
                    self.enemies.remove(proj["target"])
                    self.reward_this_step += 0.1
                    self.resources += 5
                    # SFX: Enemy_Explode
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 30),
                'color': color
            })

    def _check_termination(self):
        if self.game_over:  # Win condition met
            self.reward_this_step += 100
            return True
        if self.base_health <= 0:
            self.reward_this_step -= 100
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw base
        base_rect = pygame.Rect(self.base_pos[0]-15, self.base_pos[1]-15, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=3)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 25, self.COLOR_BASE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 25, self.COLOR_BASE_GLOW)

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.draw.circle(self.screen, spec["color"], pos, 10)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos, 7)
            pygame.draw.circle(self.screen, spec["color"], pos, 4)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_ENEMY)
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_pct = enemy["health"] / enemy["max_health"]
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0] - 10, pos[1] - 15, 20 * health_pct, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.line(self.screen, proj["color"], proj["start_pos"], pos, 2)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 3)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = max(0, min(255, int(p['life'] * 15)))
            color = p['color'] + (alpha,)
            try:
                # Use a surface for proper alpha blending
                s = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, 2, 2, 2, color)
                self.screen.blit(s, (pos[0] - 2, pos[1] - 2))
            except (TypeError, ValueError):
                # Fallback for potential color issues
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, p['color'])

    def _render_ui(self):
        # Draw top UI panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, panel_rect)

        # Base Health
        health_text = self.font_m.render(f"Base HP: {max(0, int(self.base_health))}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 8))

        # Resources
        resource_text = self.font_m.render(f"Resources: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (220, 8))

        # Wave
        wave_text = self.font_m.render(f"Wave: {self.current_wave}/{self.total_waves}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (450, 8))

        # Draw cursor and tower info only during intermission
        if self.game_phase == self.PHASE_INTERMISSION:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            cursor_pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))

            # Draw range indicator
            is_valid = self._is_valid_placement(self.cursor_pos) and self.resources >= spec["cost"]
            cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
            pygame.gfxdraw.aacircle(self.screen, cursor_pos_int[0], cursor_pos_int[1], spec["range"], cursor_color)
            pygame.gfxdraw.filled_circle(self.screen, cursor_pos_int[0], cursor_pos_int[1], spec["range"], cursor_color)

            # Draw cursor
            pygame.draw.circle(self.screen, spec["color"], cursor_pos_int, 10)

            # Draw "Start Wave" button
            pygame.draw.rect(self.screen, (0, 100, 200), self.start_wave_button, border_radius=5)
            start_text = self.font_m.render("START WAVE", True, self.COLOR_TEXT)
            self.screen.blit(start_text, (self.start_wave_button.centerx - start_text.get_width() // 2, self.start_wave_button.centery - start_text.get_height() // 2))

            # Draw Tower Info Panel
            info_panel_rect = pygame.Rect(10, self.SCREEN_HEIGHT - 60, 250, 50)
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, info_panel_rect, border_radius=5)
            name_text = self.font_s.render(f"{spec['name']} (Cost: ${spec['cost']})", True, self.COLOR_TEXT)
            stats_text = self.font_s.render(f"DMG: {spec['damage']} | Range: {spec['range']} | Rate: {spec['fire_rate']}s", True, self.COLOR_TEXT)
            self.screen.blit(name_text, (20, self.SCREEN_HEIGHT - 55))
            self.screen.blit(stats_text, (20, self.SCREEN_HEIGHT - 35))

        # Game Over / Win Text
        if self.game_over:
            if self.base_health > 0:
                text = self.font_l.render("VICTORY!", True, self.COLOR_BASE)
            else:
                text = self.font_l.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(text, (self.SCREEN_WIDTH//2 - text.get_width()//2, self.SCREEN_HEIGHT//2 - text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
        }

    def _generate_path(self):
        path = []
        x, y = 0, self.np_random.integers(100, self.SCREEN_HEIGHT - 100)
        path.append((x, y))

        while x < self.SCREEN_WIDTH - 100:
            x += self.np_random.integers(80, 150)
            x = min(x, self.SCREEN_WIDTH - 40)
            path.append((x, y))

            if x < self.SCREEN_WIDTH - 100:
                ny = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
                path.append((x, ny))
                y = ny

        path.append(self.base_pos)
        return path

    def _dist_to_segment_sq(self, p, v, w):
        l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
        if l2 == 0: return (p[0] - v[0])**2 + (p[1] - v[1])**2
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return (p[0] - projection[0])**2 + (p[1] - projection[1])**2

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This function validates that the environment conforms to the expected API.
        # The original code had a bug where it tried to get an observation
        # before the environment was reset, causing an AttributeError.
        # The fix is to follow the Gymnasium API: reset() must be called first.
        
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # 1. Reset the environment to get an initial state and observation.
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # 2. Now that the state is initialized, test the observation function.
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # 3. Test the step function with a random action.
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It is not part of the required environment implementation
    # and will not be graded.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    running = True
    while running:
        # --- Action gathering for human play ---
        movement = 0  # none
        space_pressed = 0
        shift_pressed = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        # Use get_keydown for single presses for space/shift
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_SPACE:
                    space_pressed = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()