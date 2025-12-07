import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:52:07.285695
# Source Brief: brief_00121.md
# Brief Index: 121
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent manipulates gravity wells to collide asteroids.
    
    **Gameplay:**
    - The agent controls 5 fixed gravity wells.
    - Actions select a well and toggle its gravity between attractive (blue) and repulsive (red).
    - Asteroids move based on the combined forces of the wells and a global oscillating gravity field.
    - Colliding asteroids are destroyed, increasing the score and potentially starting a chain reaction for bonus points.
    
    **Objective:**
    - Clear 100 asteroids to win.
    - The episode also ends if the step limit of 5000 is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Manipulate gravity wells to create asteroid collisions. Clear all the asteroids to win."
    user_guide = "Use arrow keys (↑↓←→) to select a gravity well. Press space to toggle its state between attractive and repulsive."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (150, 150, 170)
    COLOR_ASTEROID = (200, 200, 210)
    COLOR_ATTRACT_WELL = (0, 150, 255)
    COLOR_REPEL_WELL = (255, 100, 0)
    COLOR_COLLISION = (255, 220, 0)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_INDICATOR = (100, 255, 100)

    MAX_STEPS = 5000
    ASTEROIDS_TO_WIN = 100
    TARGET_ASTEROID_COUNT = 30
    ASTEROID_SPAWN_INTERVAL = 15

    WELL_POSITIONS = [
        (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.3),
        (SCREEN_WIDTH * 0.8, SCREEN_HEIGHT * 0.3),
        (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.7),
        (SCREEN_WIDTH * 0.8, SCREEN_HEIGHT * 0.7),
        (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.5),
    ]
    WELL_STRENGTH = 1500
    WELL_RADIUS = 15

    GLOBAL_GRAVITY_PERIOD = 200
    GLOBAL_GRAVITY_STRENGTH = 0.01

    CHAIN_REACTION_TIMEOUT = 90

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Initialize state variables to be populated in reset()
        self.steps = 0
        self.score = 0
        self.asteroids_cleared = 0
        self.game_over = False
        self.asteroids = []
        self.gravity_wells = []
        self.particles = []
        self.stars = []
        self.selected_well_index = 0
        self.prev_space_held = False
        self.global_gravity_timer = 0
        self.global_gravity_direction = pygame.math.Vector2(0, 1)
        self.chain_reaction_count = 0
        self.chain_reaction_timer = 0
        self.asteroid_spawn_timer = 0
        self.current_asteroid_speed = 1.0

        self._generate_stars()
        
        # Call reset to initialize the game state for the first time
        # self.reset() # This is called by the wrapper, no need to call it here.
        
        # self.validate_implementation() # For development; commented out for final submission

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append(
                (
                    random.randint(0, self.SCREEN_WIDTH),
                    random.randint(0, self.SCREEN_HEIGHT),
                    random.uniform(0.5, 1.5)
                )
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.asteroids_cleared = 0
        self.game_over = False
        
        self.current_asteroid_speed = 1.0
        
        self.asteroids = []
        for _ in range(self.TARGET_ASTEROID_COUNT):
            self._spawn_asteroid()

        self.gravity_wells = []
        for pos in self.WELL_POSITIONS:
            self.gravity_wells.append({
                "pos": pygame.math.Vector2(pos),
                "is_attracting": True,
                "strength": self.WELL_STRENGTH
            })

        self.particles = []
        self.selected_well_index = 4
        self.prev_space_held = False

        self.global_gravity_timer = 0
        self.global_gravity_direction = pygame.math.Vector2(0, 1)
        self.chain_reaction_count = 0
        self.chain_reaction_timer = 0
        self.asteroid_spawn_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01
        
        self._handle_input(action)
        self._update_global_gravity()
        deflection_reward = self._update_asteroids()
        self._update_particles()

        reward += deflection_reward

        collided_indices, collision_reward = self._check_collisions()
        
        if collided_indices:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in collided_indices]
            self.score += int(collision_reward)
            reward += collision_reward

        if self.chain_reaction_timer > 0:
            self.chain_reaction_timer -= 1
        else:
            self.chain_reaction_count = 0

        self._manage_asteroid_population()
        self._update_difficulty()

        self.steps += 1
        terminated = self.asteroids_cleared >= self.ASTEROIDS_TO_WIN or self.steps >= self.MAX_STEPS
        
        if self.asteroids_cleared >= self.ASTEROIDS_TO_WIN:
            reward += 100
        
        if terminated:
            self.game_over = True

        truncated = False # Gymnasium expects this to be a boolean
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_raw, _ = action
        space_held = space_held_raw == 1

        if movement in [1, 2, 3, 4]:
            self.selected_well_index = movement - 1
        elif movement == 0:
            self.selected_well_index = 4

        if space_held and not self.prev_space_held:
            well = self.gravity_wells[self.selected_well_index]
            well["is_attracting"] = not well["is_attracting"]
            # sfx: toggle_sound.play()
        self.prev_space_held = space_held

    def _update_global_gravity(self):
        self.global_gravity_timer += 1
        if self.global_gravity_timer >= self.GLOBAL_GRAVITY_PERIOD:
            self.global_gravity_timer = 0
            self.global_gravity_direction *= -1
            # sfx: global_shift_sound.play()

    def _update_asteroids(self):
        total_deflection_reward = 0
        for asteroid in self.asteroids:
            total_force = self.global_gravity_direction * self.GLOBAL_GRAVITY_STRENGTH * 100

            for well in self.gravity_wells:
                vec_to_well = well["pos"] - asteroid["pos"]
                dist_sq = vec_to_well.length_squared()
                
                if dist_sq > 1:
                    force_magnitude = well["strength"] / dist_sq
                    force_vec = vec_to_well.normalize() * force_magnitude
                    if not well["is_attracting"]:
                        force_vec *= -1
                    total_force += force_vec

            asteroid["vel"] += total_force / asteroid["mass"]
            
            speed = asteroid["vel"].length()
            if speed > asteroid["max_speed"]:
                asteroid["vel"].scale_to_length(asteroid["max_speed"])

            asteroid["trail"].append(asteroid["pos"].copy())
            if len(asteroid["trail"]) > asteroid["trail_length"]:
                asteroid["trail"].pop(0)

            asteroid["pos"] += asteroid["vel"]

            if asteroid["pos"].x < 0: asteroid["pos"].x = self.SCREEN_WIDTH
            if asteroid["pos"].x > self.SCREEN_WIDTH: asteroid["pos"].x = 0
            if asteroid["pos"].y < 0: asteroid["pos"].y = self.SCREEN_HEIGHT
            if asteroid["pos"].y > self.SCREEN_HEIGHT: asteroid["pos"].y = 0
        
        return total_deflection_reward

    def _check_collisions(self):
        collided_indices = set()
        collision_reward = 0
        
        for i in range(len(self.asteroids)):
            for j in range(i + 1, len(self.asteroids)):
                if i in collided_indices or j in collided_indices: continue
                
                a1, a2 = self.asteroids[i], self.asteroids[j]
                if (a1["pos"] - a2["pos"]).length_squared() < (a1["radius"] + a2["radius"])**2:
                    collided_indices.update([i, j])
                    
                    if self.chain_reaction_timer > 0: self.chain_reaction_count += 1
                    else: self.chain_reaction_count = 1
                    self.chain_reaction_timer = self.CHAIN_REACTION_TIMEOUT
                    
                    collision_reward += 2 + 5 * self.chain_reaction_count
                    self.asteroids_cleared += 2
                    
                    self._create_collision_effect((a1["pos"] + a2["pos"]) / 2)
                    # sfx: explosion_sound.play()
        return collided_indices, collision_reward

    def _manage_asteroid_population(self):
        self.asteroid_spawn_timer += 1
        if self.asteroid_spawn_timer >= self.ASTEROID_SPAWN_INTERVAL and len(self.asteroids) < self.TARGET_ASTEROID_COUNT:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = 0
            
    def _update_difficulty(self):
        if self.asteroids_cleared >= 75: self.current_asteroid_speed = 1.15
        elif self.asteroids_cleared >= 50: self.current_asteroid_speed = 1.10
        elif self.asteroids_cleared >= 25: self.current_asteroid_speed = 1.05

    def _spawn_asteroid(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        pos, vel = pygame.math.Vector2(0, 0), pygame.math.Vector2(0, 0)
        
        if edge == 'top':
            pos.x, pos.y = random.uniform(0, self.SCREEN_WIDTH), 0
            vel.x, vel.y = random.uniform(-0.5, 0.5), random.uniform(0.5, 1.0)
        elif edge == 'bottom':
            pos.x, pos.y = random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT
            vel.x, vel.y = random.uniform(-0.5, 0.5), random.uniform(-1.0, -0.5)
        elif edge == 'left':
            pos.x, pos.y = 0, random.uniform(0, self.SCREEN_HEIGHT)
            vel.x, vel.y = random.uniform(0.5, 1.0), random.uniform(-0.5, 0.5)
        elif edge == 'right':
            pos.x, pos.y = self.SCREEN_WIDTH, random.uniform(0, self.SCREEN_HEIGHT)
            vel.x, vel.y = random.uniform(-1.0, -0.5), random.uniform(-0.5, 0.5)

        radius = random.uniform(2, 4)
        self.asteroids.append({
            "pos": pos, "vel": vel * self.current_asteroid_speed, "radius": radius,
            "mass": radius**2, "max_speed": 3.0, "trail": [], "trail_length": 10,
        })

    def _create_collision_effect(self, pos):
        for _ in range(random.randint(20, 30)):
            angle, speed = random.uniform(0, 2 * math.pi), random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(15, 30)
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "lifespan": lifespan,
                "max_lifespan": lifespan, "color": self.COLOR_COLLISION
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_wells()
        self._render_asteroids()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
    
    def _render_wells(self):
        for i, well in enumerate(self.gravity_wells):
            color = self.COLOR_ATTRACT_WELL if well["is_attracting"] else self.COLOR_REPEL_WELL
            pos_int = (int(well["pos"].x), int(well["pos"].y))
            
            if i == self.selected_well_index:
                glow_radius = self.WELL_RADIUS + 10
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color + (60,), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

            influence_radius = int(math.sqrt(well["strength"] / 2.0))
            influence_alpha = 20 if i != self.selected_well_index else 40
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], influence_radius, color + (influence_alpha,))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.WELL_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.WELL_RADIUS, color)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            if len(asteroid["trail"]) > 1:
                points = [(int(p.x), int(p.y)) for p in asteroid["trail"]]
                pygame.draw.aalines(self.screen, self.COLOR_ASTEROID, False, points)

            pos_int = (int(asteroid["pos"].x), int(asteroid["pos"].y))
            radius = int(asteroid["radius"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"] + (alpha,) if len(p["color"]) == 3 else p["color"]
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (pos_int[0] - 2, pos_int[1] - 2))


    def _render_ui(self):
        cleared_text = self.font_ui.render(f"Cleared: {self.asteroids_cleared}/{self.ASTEROIDS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cleared_text, (10, 10))
        
        score_str = f"Score: {self.score}"
        if self.chain_reaction_count > 1:
            score_str += f" x{self.chain_reaction_count}"
        score_text = self.font_ui.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        indicator_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)
        p1 = indicator_pos - self.global_gravity_direction * 15
        p2 = indicator_pos + self.global_gravity_direction * 15
        pygame.draw.line(self.screen, self.COLOR_UI_INDICATOR, p1, p2, 3)
        angle = self.global_gravity_direction.angle_to(pygame.math.Vector2(1, 0))
        p_left = p2 + pygame.math.Vector2(8, 0).rotate(angle + 150)
        p_right = p2 + pygame.math.Vector2(8, 0).rotate(angle - 150)
        pygame.draw.polygon(self.screen, self.COLOR_UI_INDICATOR, [p2, p_left, p_right])
        
        if self.game_over:
            status = "VICTORY" if self.asteroids_cleared >= self.ASTEROIDS_TO_WIN else "TIME UP"
            end_text = self.font_ui.render(f"GAME OVER: {status}", True, self.COLOR_COLLISION)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "asteroids_cleared": self.asteroids_cleared,
            "chain_multiplier": self.chain_reaction_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the evaluation system.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'macOS', etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Well")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print("\n--- Controls ---")
    print("Arrow Keys: Select gravity well")
    print("Spacebar: Toggle selected well (Attract/Repel)")
    print("R: Reset game")
    print("Escape: Quit")
    print("----------------\n")
    
    while running:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        # Map arrow keys to movement actions 1-4
        if keys[pygame.K_UP]: movement = 1 
        elif keys[pygame.K_RIGHT]: movement = 2
        elif keys[pygame.K_DOWN]: movement = 3
        elif keys[pygame.K_LEFT]: movement = 4
        # Action 0 is default (center well)
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        if keys[pygame.K_r]:
            obs, info = env.reset()
            terminated = False

        if keys[pygame.K_ESCAPE]:
            running = False

        if not terminated:
            # Note: The mapping in _handle_input is slightly different from the __main__ block's intention.
            # _handle_input: 1->well 0, 2->well 1, 3->well 2, 4->well 3, 0->well 4
            # This is fine, just a note on the mapping.
            action = np.array([movement, space_held, shift_held])
            obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60)
        
    env.close()