import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:11:28.772262
# Source Brief: brief_01545.md
# Brief Index: 1545
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central core from incoming enemies by controlling a trio of orbiting guardians. "
        "Adjust your guardians' size to balance speed and power."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected guardian. "
        "Hold space to increase its size and shift to decrease it."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_CORE = (255, 255, 255)
    COLOR_GUARDIAN = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SELECTION = (255, 255, 0)

    # Core
    CORE_MAX_HEALTH = 1000
    CORE_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    CORE_BASE_RADIUS = 20

    # Guardians
    NUM_GUARDIANS = 3
    GUARDIAN_MIN_SIZE = 8
    GUARDIAN_MAX_SIZE = 40
    GUARDIAN_SIZE_STEP = 1.0
    GUARDIAN_NUDGE_FORCE = 0.8
    GUARDIAN_FRICTION = 0.95
    GUARDIAN_SELECTION_CYCLE = 15 # Steps to cycle selected guardian

    # Enemies
    ENEMY_BASE_HEALTH = 10
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPAWN_RATE_INITIAL = 90 # Steps between spawns
    ENEMY_SPAWN_RATE_INCREASE = 0.995 # Multiplier every 100 steps
    ENEMY_HEALTH_INCREASE = 1.01 # Multiplier every 200 steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.core_health = 0
        self.guardians = []
        self.enemies = []
        self.particles = []
        self.selected_guardian_idx = 0
        self.selection_timer = 0
        self.enemy_spawn_timer = 0
        self.current_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.current_enemy_health = self.ENEMY_BASE_HEALTH
        
        # --- Pre-rendered Background ---
        self._nebula_bg = self._create_nebula_background()

        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # Validation is not needed in the final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.core_health = self.CORE_MAX_HEALTH
        self.guardians = self._create_initial_guardians()
        self.enemies = []
        self.particles = []

        self.selected_guardian_idx = 0
        self.selection_timer = 0
        self.enemy_spawn_timer = 0
        self.current_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.current_enemy_health = self.ENEMY_BASE_HEALTH

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Handle Input and Timers ---
        self._handle_input(action)
        self._update_timers()

        # --- Update Game Logic ---
        self._update_guardians()
        self._update_enemies()
        self._update_particles()
        
        # --- Handle Collisions and Damage ---
        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        # --- Spawn New Enemies ---
        self._spawn_enemies()

        # --- Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and self.core_health <= 0:
            self.game_over = True
            # No penalty for losing, just absence of survival reward
        elif truncated:
            reward += 100 # Victory reward
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.blit(self._nebula_bg, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "core_health": self.core_health}

    # --- Helper Methods for Initialization ---

    def _create_initial_guardians(self):
        guardians = []
        for i in range(self.NUM_GUARDIANS):
            angle = (2 * math.pi / self.NUM_GUARDIANS) * i
            dist = 80
            pos = pygame.math.Vector2(
                self.CORE_POS[0] + dist * math.cos(angle),
                self.CORE_POS[1] + dist * math.sin(angle),
            )
            guardians.append({
                "pos": pos,
                "vel": pygame.math.Vector2(0, 0),
                "size": self.GUARDIAN_MIN_SIZE + (self.GUARDIAN_MAX_SIZE - self.GUARDIAN_MIN_SIZE) / 2,
            })
        return guardians

    def _create_nebula_background(self):
        bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        bg_surface.fill(self.COLOR_BG)
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.randint(1, 4)
            color_val = random.randint(20, 50)
            color = (color_val, color_val, color_val + 20)
            pygame.gfxdraw.filled_circle(bg_surface, x, y, size, color)
        return bg_surface

    # --- Helper Methods for Step Logic ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        g = self.guardians[self.selected_guardian_idx]
        
        # Apply movement nudge
        if movement == 1: g["vel"].y -= self.GUARDIAN_NUDGE_FORCE
        elif movement == 2: g["vel"].y += self.GUARDIAN_NUDGE_FORCE
        elif movement == 3: g["vel"].x -= self.GUARDIAN_NUDGE_FORCE
        elif movement == 4: g["vel"].x += self.GUARDIAN_NUDGE_FORCE

        # Apply size change
        if space_held:
            g["size"] = min(self.GUARDIAN_MAX_SIZE, g["size"] + self.GUARDIAN_SIZE_STEP)
        if shift_held:
            g["size"] = max(self.GUARDIAN_MIN_SIZE, g["size"] - self.GUARDIAN_SIZE_STEP)

    def _update_timers(self):
        self.selection_timer += 1
        if self.selection_timer >= self.GUARDIAN_SELECTION_CYCLE:
            self.selection_timer = 0
            self.selected_guardian_idx = (self.selected_guardian_idx + 1) % self.NUM_GUARDIANS
        
        self.enemy_spawn_timer += 1
        
        if self.steps > 0 and self.steps % 100 == 0:
            self.current_spawn_rate *= self.ENEMY_SPAWN_RATE_INCREASE
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_enemy_health *= self.ENEMY_HEALTH_INCREASE

    def _update_guardians(self):
        for g in self.guardians:
            # Inverse speed based on size
            speed_modifier = 1 - ((g["size"] - self.GUARDIAN_MIN_SIZE) / (self.GUARDIAN_MAX_SIZE - self.GUARDIAN_MIN_SIZE))
            speed_modifier = 0.5 + speed_modifier # Ensure it doesn't go to 0
            
            g["pos"] += g["vel"] * speed_modifier
            g["vel"] *= self.GUARDIAN_FRICTION
            
            # Boundary checks
            g["pos"].x = np.clip(g["pos"].x, g["size"], self.SCREEN_WIDTH - g["size"])
            g["pos"].y = np.clip(g["pos"].y, g["size"], self.SCREEN_HEIGHT - g["size"])

    def _update_enemies(self):
        enemies_to_remove = []
        for i, e in enumerate(self.enemies):
            e["pos"] += e["vel"]
            if not (0 < e["pos"].x < self.SCREEN_WIDTH and 0 < e["pos"].y < self.SCREEN_HEIGHT):
                enemies_to_remove.append(i)
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _handle_collisions(self):
        reward = 0
        
        # Guardian vs Enemy
        enemies_to_remove = []
        for i, e in enumerate(self.enemies):
            for g in self.guardians:
                dist_vec = e["pos"] - g["pos"]
                if dist_vec.length_squared() < (e["size"] + g["size"])**2:
                    # Damage enemy
                    damage = (g["size"] / self.GUARDIAN_MAX_SIZE) * 20
                    e["health"] -= damage
                    reward += 1 # Reward for damaging
                    
                    # Repel effect
                    if dist_vec.length() > 0:
                        repel_force = dist_vec.normalize() * 3
                        e["vel"] += repel_force
                        g["vel"] -= repel_force * 0.5
                    
                    # Create impact particles
                    self._create_particles(e["pos"], 5, self.COLOR_ENEMY, 2)
                    
                    if e["health"] <= 0 and i not in enemies_to_remove:
                        reward += 5 # Reward for destroying
                        enemies_to_remove.append(i)
                        # Sound: Enemy_Destroyed.wav
                        self._create_particles(e["pos"], 20, self.COLOR_ENEMY, 4)

        # Enemy vs Core
        core_pos_vec = pygame.math.Vector2(self.CORE_POS)
        core_radius = self.CORE_BASE_RADIUS * (self.core_health / self.CORE_MAX_HEALTH)
        for i, e in enumerate(self.enemies):
            if i in enemies_to_remove: continue
            dist_vec = e["pos"] - core_pos_vec
            if dist_vec.length_squared() < (e["size"] + core_radius)**2:
                self.core_health -= e["damage"]
                enemies_to_remove.append(i)
                # Sound: Core_Hit.wav
                self._create_particles(core_pos_vec, 15, self.COLOR_CORE, 3)

        # Remove destroyed enemies
        for i in sorted(list(set(enemies_to_remove)), reverse=True):
            if i < len(self.enemies):
                del self.enemies[i]
        
        # Guardian vs Guardian
        for i, g1 in enumerate(self.guardians):
            for j, g2 in enumerate(self.guardians):
                if i >= j: continue
                dist_vec = g1["pos"] - g2["pos"]
                if dist_vec.length_squared() < (g1["size"] + g2["size"])**2:
                    if dist_vec.length() > 0:
                        overlap = (g1["size"] + g2["size"]) - dist_vec.length()
                        push_vec = dist_vec.normalize() * overlap * 0.5
                        g1["pos"] += push_vec
                        g2["pos"] -= push_vec

        return reward

    def _spawn_enemies(self):
        if self.enemy_spawn_timer >= self.current_spawn_rate:
            self.enemy_spawn_timer = 0
            
            edge = random.choice(["top", "bottom", "left", "right"])
            if edge == "top":
                pos = pygame.math.Vector2(random.randint(0, self.SCREEN_WIDTH), -10)
            elif edge == "bottom":
                pos = pygame.math.Vector2(random.randint(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10)
            elif edge == "left":
                pos = pygame.math.Vector2(-10, random.randint(0, self.SCREEN_HEIGHT))
            else: # right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + 10, random.randint(0, self.SCREEN_HEIGHT))

            direction = (pygame.math.Vector2(self.CORE_POS) - pos).normalize()
            vel = direction * self.ENEMY_BASE_SPEED
            
            self.enemies.append({
                "pos": pos,
                "vel": vel,
                "health": self.current_enemy_health,
                "size": 10,
                "damage": 50,
            })

    def _check_termination(self):
        return self.core_health <= 0

    # --- Rendering Methods ---

    def _render_game(self):
        # Particles
        for p in self.particles:
            size = int(p["size"] * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, p["color"])

        # Core
        self._render_core()

        # Enemies
        for e in self.enemies:
            p1 = e["pos"] + pygame.math.Vector2(0, -e["size"]).rotate_rad(self.steps * 0.1)
            p2 = e["pos"] + pygame.math.Vector2(e["size"] * 0.866, e["size"] * 0.5).rotate_rad(self.steps * 0.1)
            p3 = e["pos"] + pygame.math.Vector2(-e["size"] * 0.866, e["size"] * 0.5).rotate_rad(self.steps * 0.1)
            points = [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        
        # Guardians
        for i, g in enumerate(self.guardians):
            self._render_glowing_circle(self.screen, g["pos"], g["size"], self.COLOR_GUARDIAN)
            if i == self.selected_guardian_idx and not self.game_over:
                self._render_selection_indicator(g)

    def _render_core(self):
        health_ratio = max(0, self.core_health / self.CORE_MAX_HEALTH)
        current_radius = int(self.CORE_BASE_RADIUS * health_ratio)
        
        # Health bar circle
        if health_ratio > 0:
            pygame.gfxdraw.aacircle(self.screen, self.CORE_POS[0], self.CORE_POS[1], self.CORE_BASE_RADIUS, (50, 50, 80))
            end_angle = 360 * health_ratio
            if end_angle > 0:
                rect = (self.CORE_POS[0] - self.CORE_BASE_RADIUS, self.CORE_POS[1] - self.CORE_BASE_RADIUS, self.CORE_BASE_RADIUS*2, self.CORE_BASE_RADIUS*2)
                pygame.draw.arc(self.screen, self.COLOR_CORE, rect, math.radians(90), math.radians(90 + end_angle), 2)
        
        # Core itself
        if current_radius > 0:
            self._render_glowing_circle(self.screen, pygame.math.Vector2(self.CORE_POS), current_radius, self.COLOR_CORE)

    def _render_selection_indicator(self, guardian):
        angle = (self.steps * 6) % 360
        for i in range(3):
            current_angle = math.radians(angle + i * 120)
            dist = guardian["size"] + 8 + 3 * math.sin(math.radians(self.steps * 4))
            x = int(guardian["pos"].x + dist * math.cos(current_angle))
            y = int(guardian["pos"].y + dist * math.sin(current_angle))
            pygame.gfxdraw.filled_circle(self.screen, x, y, 2, self.COLOR_SELECTION)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_ui.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.core_health <= 0:
                msg = "CORE DESTROYED"
            else:
                msg = "VICTORY"
            over_text = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(over_text, text_rect)

    def _render_glowing_circle(self, surface, pos, radius, color):
        radius = int(radius)
        if radius <= 0: return
        
        x, y = int(pos.x), int(pos.y)
        
        for i in range(radius, 0, -2):
            alpha = int(180 * (1 - (i / radius))**2)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, x, y, i + (radius - i)//2, glow_color)
            
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)

    def _create_particles(self, pos, count, color, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_scale
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(10, 25)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
                "size": random.randint(2, 5)
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and debugging
    # It will not be executed by the evaluation environment,
    # but it's useful for testing the game.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Inner Mind Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # To hold down keys
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }
    movement_action = 0
    space_action = 0
    shift_action = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_map: movement_action = key_map[event.key]
                if event.key == pygame.K_SPACE: space_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False
            if event.type == pygame.KEYUP:
                if event.key in key_map and movement_action == key_map[event.key]:
                    movement_action = 0
                if event.key == pygame.K_SPACE: space_action = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 0

        if terminated or truncated:
            # You can add a delay or a 'press key to restart' message here
            continue

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()
    print(f"Game Over. Final Score: {total_reward:.2f}")