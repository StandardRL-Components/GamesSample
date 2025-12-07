import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:49:37.796642
# Source Brief: brief_00108.md
# Brief Index: 108
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player protects ancient trees from shadowy entities.

    The player controls three magical wards, which can be moved to repel enemies.
    The goal is to survive 20 waves of increasingly difficult enemies.
    After every 5 waves, the player can upgrade a ward's power.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Space button (0: released, 1: held) - Press to cycle selected ward
    - action[2]: Shift button (0: released, 1: held) - Press to confirm upgrade

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.

    Reward Structure:
    - +0.1 for each enemy repelled by a ward per step.
    - -0.01 for each step an enemy is within a danger radius of a tree.
    - +1 for completing a wave.
    - +100 for winning the game (surviving all waves).
    - -100 for losing the game (all trees destroyed).
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Protect ancient trees from waves of shadowy entities by positioning magical wards to repel them."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected ward. Press space to cycle which ward is selected. "
        "Use shift to confirm an upgrade."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000 # Increased to allow for 20 waves

    # Colors
    COLOR_BG = (5, 15, 25)
    COLOR_BG_ACCENT = (15, 35, 45)
    COLOR_TREE = (40, 200, 80)
    COLOR_TREE_GLOW = (40, 200, 80, 50)
    COLOR_WARD = (60, 180, 255)
    COLOR_WARD_GLOW = (60, 180, 255, 60)
    COLOR_WARD_SELECTED = (255, 255, 100)
    COLOR_ENEMY = (10, 0, 0)
    COLOR_ENEMY_OUTLINE = (200, 30, 30)
    COLOR_PARTICLE = (80, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 40, 60, 180)
    COLOR_UPGRADE_SELECT = (255, 255, 0, 150)

    # Game Parameters
    NUM_TREES = 3
    NUM_WARDS = 3
    TREE_HEALTH_MAX = 100
    TREE_RADIUS = 25
    TREE_DANGER_RADIUS = 150
    WARD_MOVE_SPEED = 5
    ENEMY_BASE_SPEED = 0.5
    ENEMY_SPEED_WAVE_INCREMENT = 0.05
    ENEMY_BASE_COUNT = 1
    ENEMY_COUNT_WAVE_INCREMENT = 1
    TOTAL_WAVES = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.trees = []
        self.wards = []
        self.enemies = []
        self.particles = []
        self.selected_ward_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.upgrade_phase = False
        self.upgrade_selection_idx = 0
        self.bg_stars = []
        self.last_movement = 0

        self._generate_background()
        # self.reset() is called by the wrapper, no need to call it here.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed python's random module
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.selected_ward_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.upgrade_phase = False
        self.upgrade_selection_idx = 0
        self.last_movement = 0

        # --- Initialize Game Elements ---
        self.trees = [
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT / 2),
                "health": self.TREE_HEALTH_MAX,
            },
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT / 2),
                "health": self.TREE_HEALTH_MAX,
            },
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT / 2),
                "health": self.TREE_HEALTH_MAX,
            },
        ]

        self.wards = [
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.25),
                "strength": 2.0,
                "radius": 50,
            },
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT * 0.25),
                "strength": 2.0,
                "radius": 50,
            },
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT * 0.25),
                "strength": 2.0,
                "radius": 50,
            },
        ]

        self.enemies = []
        self.particles = []
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self._check_termination()
        if self.game_over:
            if all(tree["health"] <= 0 for tree in self.trees):
                reward -= 100 # Defeat
            elif self.wave > self.TOTAL_WAVES:
                reward += 100 # Victory
            return self._get_observation(), reward, self.game_over, False, self._get_info()

        # --- Handle Input ---
        self._handle_input(action)

        if self.upgrade_phase:
            # Game is paused, waiting for player to upgrade
            pass
        else:
            # --- Update Game Logic ---
            reward += self._update_enemies()
            self._update_particles()
            self._check_collisions()
            self.steps += 1

            # --- Check for Wave Completion ---
            if not self.enemies:
                reward += 1.0 # Wave complete bonus
                self.wave += 1

                if self.wave > self.TOTAL_WAVES:
                    self.game_over = True
                elif (self.wave -1) % 5 == 0 and self.wave > 1:
                    self.upgrade_phase = True
                    self.upgrade_selection_idx = 0
                    # Sound effect: upgrade_phase_start.wav
                else:
                    self._spawn_wave()
                    # Sound effect: new_wave.wav
        
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Private Helper Methods ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Ward Selection (Spacebar) ---
        if space_held and not self.prev_space_held:
            self.selected_ward_idx = (self.selected_ward_idx + 1) % self.NUM_WARDS
            # Sound effect: ward_select.wav

        # --- Upgrade Phase Logic ---
        if self.upgrade_phase:
            if shift_held and not self.prev_shift_held:
                # Upgrade selected ward
                ward = self.wards[self.upgrade_selection_idx]
                ward["strength"] += 1.0
                ward["radius"] += 10
                self.upgrade_phase = False
                self._spawn_wave()
                # Sound effect: upgrade_confirm.wav
            else:
                # Cycle selection with movement keys
                if movement != 0 and movement != self.last_movement:
                    if movement in [1, 3]: # Up or Left
                        self.upgrade_selection_idx = (self.upgrade_selection_idx - 1 + self.NUM_WARDS) % self.NUM_WARDS
                    elif movement in [2, 4]: # Down or Right
                        self.upgrade_selection_idx = (self.upgrade_selection_idx + 1) % self.NUM_WARDS
        else:
            # --- Ward Movement ---
            ward = self.wards[self.selected_ward_idx]
            if movement == 1: ward["pos"].y -= self.WARD_MOVE_SPEED # Up
            if movement == 2: ward["pos"].y += self.WARD_MOVE_SPEED # Down
            if movement == 3: ward["pos"].x -= self.WARD_MOVE_SPEED # Left
            if movement == 4: ward["pos"].x += self.WARD_MOVE_SPEED # Right

            # Clamp ward position to screen bounds
            ward["pos"].x = max(0, min(self.SCREEN_WIDTH, ward["pos"].x))
            ward["pos"].y = max(0, min(self.SCREEN_HEIGHT, ward["pos"].y))

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.last_movement = movement

    def _update_enemies(self):
        step_reward = 0
        living_trees = [t for t in self.trees if t["health"] > 0]
        if not living_trees:
            return 0

        for enemy in self.enemies:
            # Find closest living tree
            closest_tree = min(living_trees, key=lambda t: enemy["pos"].distance_to(t["pos"]))
            
            # Move towards tree
            direction = (closest_tree["pos"] - enemy["pos"]).normalize()
            enemy["pos"] += direction * enemy["speed"]

            # Pulsate for visual effect
            enemy["pulse"] = (enemy["pulse"] + 0.2) % (2 * math.pi)
            
            # Penalty for being near a tree
            if enemy["pos"].distance_to(closest_tree["pos"]) < self.TREE_DANGER_RADIUS:
                step_reward -= 0.01

            # Check for ward repulsion
            for ward in self.wards:
                dist_vec = enemy["pos"] - ward["pos"]
                distance = dist_vec.length()
                if distance > 0 and distance < ward["radius"]:
                    repulsion_strength = ward["strength"] * (1 - (distance / ward["radius"]))
                    enemy["pos"] += dist_vec.normalize() * repulsion_strength
                    step_reward += 0.1 # Reward for active repulsion

                    # Spawn particles
                    if random.random() < 0.3:
                        self._spawn_particles(ward["pos"], 2)
        return step_reward

    def _check_collisions(self):
        for enemy in self.enemies[:]:
            for tree in self.trees:
                if tree["health"] > 0:
                    if enemy["pos"].distance_to(tree["pos"]) < self.TREE_RADIUS:
                        tree["health"] -= 10
                        self.enemies.remove(enemy)
                        # Sound effect: tree_damage.wav
                        self._spawn_particles(tree["pos"], 10, self.COLOR_ENEMY_OUTLINE)
                        break

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        self.enemies.clear()
        num_enemies = self.ENEMY_BASE_COUNT + (self.wave - 1) * self.ENEMY_COUNT_WAVE_INCREMENT
        enemy_speed = self.ENEMY_BASE_SPEED + (self.wave - 1) * self.ENEMY_SPEED_WAVE_INCREMENT
        
        for _ in range(num_enemies):
            # Spawn on screen edges
            edge = random.choice(["top", "bottom", "left", "right"])
            if edge == "top": pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), -20)
            elif edge == "bottom": pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
            elif edge == "left": pos = pygame.Vector2(-20, random.uniform(0, self.SCREEN_HEIGHT))
            else: pos = pygame.Vector2(self.SCREEN_WIDTH + 20, random.uniform(0, self.SCREEN_HEIGHT))
            
            # Create blob shape
            num_points = 8
            points = []
            for i in range(num_points):
                angle = (i / num_points) * 2 * math.pi
                radius = random.uniform(8, 15)
                points.append((math.cos(angle) * radius, math.sin(angle) * radius))

            self.enemies.append({"pos": pos, "speed": enemy_speed, "shape_points": points, "pulse": random.uniform(0, 2*math.pi)})

    def _spawn_particles(self, pos, count, color=None):
        if color is None: color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": random.randint(10, 20), "color": color})

    def _check_termination(self):
        if all(tree["health"] <= 0 for tree in self.trees):
            return True
        if self.wave > self.TOTAL_WAVES:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "tree_healths": [t["health"] for t in self.trees],
            "ward_strengths": [w["strength"] for w in self.wards],
        }

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_trees()
        self._render_wards()
        self._render_enemies()
        self._render_ui()
        if self.upgrade_phase:
            self._render_upgrade_screen()
        if self.game_over:
            self._render_game_over()

    def _generate_background(self):
        self.bg_stars = []
        for _ in range(100):
            self.bg_stars.append(
                (
                    (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                    random.randint(1, 2)
                )
            )

    def _render_background(self):
        for pos, size in self.bg_stars:
            pygame.draw.circle(self.screen, self.COLOR_BG_ACCENT, pos, size)

    def _render_trees(self):
        for tree in self.trees:
            if tree["health"] > 0:
                pos = (int(tree["pos"].x), int(tree["pos"].y))
                # Glow
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TREE_RADIUS + 5, self.COLOR_TREE_GLOW)
                # Tree
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TREE_RADIUS, self.COLOR_TREE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TREE_RADIUS, self.COLOR_TREE)
                # Health bar
                health_pct = max(0, tree["health"] / self.TREE_HEALTH_MAX)
                bar_width = 50
                bar_height = 5
                bar_pos = (pos[0] - bar_width // 2, pos[1] + self.TREE_RADIUS + 5)
                pygame.draw.rect(self.screen, (255,0,0), (*bar_pos, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0,255,0), (*bar_pos, int(bar_width * health_pct), bar_height))

    def _render_wards(self):
        for i, ward in enumerate(self.wards):
            pos = (int(ward["pos"].x), int(ward["pos"].y))
            radius = int(ward["radius"])
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_WARD_GLOW)
            # Ward core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_WARD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_WARD)
            # Selected highlight
            if i == self.selected_ward_idx and not self.upgrade_phase:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_WARD_SELECTED)

    def _render_enemies(self):
        for enemy in self.enemies:
            pulse_offset = math.sin(enemy["pulse"]) * 2
            
            # Create base points relative to (0,0)
            base_points = [(p[0], p[1]) for p in enemy["shape_points"]]
            
            # Scale points for pulse effect
            scaled_points = []
            for p_x, p_y in base_points:
                vec = pygame.Vector2(p_x, p_y)
                scaled_vec = vec * (1 + pulse_offset * 0.1)
                # Translate to enemy position
                scaled_points.append((enemy["pos"].x + scaled_vec.x, enemy["pos"].y + scaled_vec.y))

            if len(scaled_points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, scaled_points, self.COLOR_ENEMY)
                pygame.gfxdraw.aapolygon(self.screen, scaled_points, self.COLOR_ENEMY_OUTLINE)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20.0))
            color = (*p["color"], alpha)
            if len(color) == 4:
                # This is a crude way to handle alpha for basic shapes
                s = pygame.Surface((3,3), pygame.SRCALPHA)
                s.fill(color)
                self.screen.blit(s, (int(p["pos"].x-1), int(p["pos"].y-1)))

    def _render_ui(self):
        # Wave
        wave_text = self.font_large.render(f"WAVE {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 30))
        self.screen.blit(score_text, score_rect)
        # Tree count
        living_trees = sum(1 for t in self.trees if t["health"] > 0)
        tree_text = self.font_large.render(f"TREES: {living_trees}/{self.NUM_TREES}", True, self.COLOR_TEXT)
        tree_rect = tree_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(tree_text, tree_rect)

    def _render_upgrade_screen(self):
        # Dim background
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Text
        title_text = self.font_huge.render("CHOOSE A WARD TO EMPOWER", True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT * 0.3))
        self.screen.blit(title_text, title_rect)
        
        info_text = self.font_large.render("Use ARROWS to select, SHIFT to confirm", True, self.COLOR_TEXT)
        info_rect = info_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT * 0.7))
        self.screen.blit(info_text, info_rect)

        # Highlight selected ward for upgrade
        ward_to_upgrade = self.wards[self.upgrade_selection_idx]
        pos = (int(ward_to_upgrade["pos"].x), int(ward_to_upgrade["pos"].y))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(ward_to_upgrade["radius"] + 10), self.COLOR_UPGRADE_SELECT)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(ward_to_upgrade["radius"] + 11), self.COLOR_UPGRADE_SELECT)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if self.wave > self.TOTAL_WAVES:
            msg = "VICTORY"
            color = (100, 255, 100)
        else:
            msg = "DEFEAT"
            color = (255, 100, 100)
            
        text = self.font_huge.render(msg, True, color)
        rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    # Unset the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Whispering Woods Defense")
    clock = pygame.time.Clock()

    total_reward = 0
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            print("Press 'R' to reset.")
            # Keep running to allow reset by waiting for the 'R' key event
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(GameEnv.FPS)

    env.close()