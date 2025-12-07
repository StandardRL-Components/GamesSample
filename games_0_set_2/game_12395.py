import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:48:23.236922
# Source Brief: brief_02395.md
# Brief Index: 2395
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating the exploration of a microscopic ecosystem.
    The agent controls a nano-sub, collecting samples of various organisms
    to complete an ecological survey.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a nano-sub to explore a microscopic world, collecting samples of strange organisms to complete your survey."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire a collection projectile. Press shift to open/close the skill tree."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30  # Assumed FPS for smooth interpolation
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (10, 25, 40)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 40)
    COLOR_PROJECTILE = (150, 255, 255)
    COLOR_TRAJECTORY = (255, 255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 40, 60, 180)
    COLOR_UI_HIGHLIGHT = (100, 180, 255)

    ORGANISM_PALETTE = [
        (255, 100, 100), (255, 180, 100), (180, 255, 100),
        (100, 255, 180), (100, 180, 255), (180, 100, 255),
        (255, 100, 180), (255, 255, 100)
    ]
    TOTAL_ORGANISM_TYPES = 8

    # Player
    PLAYER_RADIUS = 12
    PLAYER_ACCEL = 0.8
    PLAYER_DRAG = 0.92

    # Projectile
    PROJECTILE_RADIUS = 4
    PROJECTILE_BASE_SPEED = 10
    PROJECTILE_BASE_LIFESPAN = 40  # in frames
    PROJECTILE_COOLDOWN = 10 # in frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 16)
        self.font_medium = pygame.font.SysFont("sans-serif", 24)
        self.font_large = pygame.font.SysFont("sans-serif", 32)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.aim_direction = None
        self.projectiles = None
        self.organisms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.projectile_cooldown_timer = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.last_movement_action = None

        # Progression state
        self.collected_types = None
        self.successful_collections = None
        self.organism_spawn_level = None
        self.current_organism_speed_modifier = None

        # Skill Tree State
        self.skill_tree_open = None
        self.skill_cursor = None
        self.skill_points = None
        self.skills = None

        # Background elements
        self._background_elements = self._create_background_elements()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.aim_direction = pygame.Vector2(1, 0)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.projectiles = []
        self.organisms = []
        self.particles = []
        self.projectile_cooldown_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_movement_action = 0

        # Progression state
        self.collected_types = set()
        self.successful_collections = 0
        self.organism_spawn_level = 1
        self.current_organism_speed_modifier = 1.0

        # Skill Tree State
        self.skill_tree_open = False
        self.skill_cursor = 0
        self.skill_points = 0
        self.skills = {
            "proj_speed": {"level": 0, "max_level": 3, "cost": 1, "name": "Projectile Speed"},
            "proj_range": {"level": 0, "max_level": 3, "cost": 1, "name": "Collection Range"},
            "analysis": {"level": 0, "max_level": 3, "cost": 2, "name": "Analysis Bonus"},
        }

        # Initial organism spawn
        for _ in range(5):
            self._spawn_organism()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- Handle Input and State Changes ---
        dist_before = self._get_dist_to_closest_uncollected()
        
        self._handle_actions(movement, space_action, shift_action)
        
        if not self.skill_tree_open:
            # --- Update Game Logic ---
            self._update_player()
            self._update_projectiles()
            self._update_organisms()
            reward += self._handle_collisions()
            self._update_particles()
            self._cleanup_objects()
            
            # --- Spawn new organisms if needed ---
            if len(self.organisms) < 5:
                self._spawn_organism()

            # --- Calculate Rewards ---
            dist_after = self._get_dist_to_closest_uncollected()
            if dist_after is not None and dist_before is not None:
                if dist_after < dist_before:
                    reward += 0.01 # Small reward for getting closer
                else:
                    reward -= 0.01 # Small penalty for moving away
            
            self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated and len(self.collected_types) == self.TOTAL_ORGANISM_TYPES:
            reward += 100 # Victory reward

        self.score += reward
        self.prev_space_held = space_action
        self.prev_shift_held = shift_action

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Helper Methods for step() ---

    def _handle_actions(self, movement, space_held, shift_held):
        # Shift action (toggles skill tree)
        if shift_held and not self.prev_shift_held:
            self.skill_tree_open = not self.skill_tree_open
            # sfx: menu_open_close

        if self.skill_tree_open:
            # Handle skill tree navigation
            if movement == 1 and self.last_movement_action != 1: # Up
                self.skill_cursor = (self.skill_cursor - 1) % len(self.skills)
                # sfx: menu_navigate
            elif movement == 2 and self.last_movement_action != 2: # Down
                self.skill_cursor = (self.skill_cursor + 1) % len(self.skills)
                # sfx: menu_navigate

            # Handle skill purchase
            if space_held and not self.prev_space_held:
                skill_key = list(self.skills.keys())[self.skill_cursor]
                skill = self.skills[skill_key]
                if self.skill_points >= skill["cost"] and skill["level"] < skill["max_level"]:
                    self.skill_points -= skill["cost"]
                    skill["level"] += 1
                    # sfx: skill_unlock
        else:
            # Handle player movement
            accel_vec = pygame.Vector2(0, 0)
            if movement == 1: accel_vec.y = -1
            elif movement == 2: accel_vec.y = 1
            elif movement == 3: accel_vec.x = -1
            elif movement == 4: accel_vec.x = 1

            if accel_vec.length() > 0:
                accel_vec.scale_to_length(self.PLAYER_ACCEL)
                self.player_vel += accel_vec
                if self.player_vel.length() > 0:
                    self.aim_direction = pygame.Vector2(self.player_vel).normalize()
            
            # Handle projectile firing
            if space_held and self.projectile_cooldown_timer == 0:
                self._fire_projectile()
        
        self.last_movement_action = movement

    def _fire_projectile(self):
        proj_speed = self.PROJECTILE_BASE_SPEED * (1 + 0.2 * self.skills["proj_speed"]["level"])
        proj_lifespan = self.PROJECTILE_BASE_LIFESPAN * (1 + 0.25 * self.skills["proj_range"]["level"])
        
        start_pos = self.player_pos + self.aim_direction * (self.PLAYER_RADIUS + 5)
        velocity = self.aim_direction * proj_speed
        
        self.projectiles.append({
            "pos": start_pos,
            "vel": velocity,
            "lifespan": proj_lifespan,
            "radius": self.PROJECTILE_RADIUS
        })
        self.projectile_cooldown_timer = self.PROJECTILE_COOLDOWN
        # sfx: shoot_projectile

    def _update_player(self):
        self.player_vel *= self.PLAYER_DRAG
        if self.player_vel.length() < 0.1:
            self.player_vel = pygame.Vector2(0, 0)
        self.player_pos += self.player_vel

        # Boundary checks
        self.player_pos.x = max(self.PLAYER_RADIUS, min(self.WIDTH - self.PLAYER_RADIUS, self.player_pos.x))
        self.player_pos.y = max(self.PLAYER_RADIUS, min(self.HEIGHT - self.PLAYER_RADIUS, self.player_pos.y))
        
        if self.projectile_cooldown_timer > 0:
            self.projectile_cooldown_timer -= 1

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _update_organisms(self):
        for o in self.organisms:
            pattern = o["pattern"]
            if pattern == "linear":
                o["pos"] += o["vel"]
                if not (0 < o["pos"].x < self.WIDTH) or not (0 < o["pos"].y < self.HEIGHT):
                    o["vel"] *= -1 # Simple bounce off screen edges
            elif pattern == "circular":
                o["pattern_state"]["angle"] += o["pattern_state"]["speed"]
                o["pos"].x = o["pattern_state"]["center"].x + math.cos(o["pattern_state"]["angle"]) * o["pattern_state"]["radius"]
                o["pos"].y = o["pattern_state"]["center"].y + math.sin(o["pattern_state"]["angle"]) * o["pattern_state"]["radius"]

    def _handle_collisions(self):
        reward = 0
        projectiles_to_remove = []
        organisms_to_remove = []

        for p_idx, p in enumerate(self.projectiles):
            for o_idx, o in enumerate(self.organisms):
                if p_idx in projectiles_to_remove or o_idx in organisms_to_remove:
                    continue
                
                dist = p["pos"].distance_to(o["pos"])
                if dist < p["radius"] + o["radius"]:
                    # Collision detected
                    # sfx: collect_sample
                    self._create_particle_burst(o["pos"], o["color"])
                    
                    reward += 1.0 # Base reward for collection
                    if o["type_id"] not in self.collected_types:
                        self.collected_types.add(o["type_id"])
                        reward += 5.0 # Bonus for new type
                        self.skill_points += 1 # Gain a skill point for new discovery
                    
                    analysis_bonus = 0.5 * self.skills["analysis"]["level"]
                    reward += analysis_bonus

                    projectiles_to_remove.append(p_idx)
                    organisms_to_remove.append(o_idx)
                    
                    self.successful_collections += 1
                    self._update_difficulty()
                    break
        
        # Remove objects in reverse index order to avoid errors
        for idx in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[idx]
        for idx in sorted(organisms_to_remove, reverse=True):
            del self.organisms[idx]
            
        return reward

    def _update_difficulty(self):
        # Increase speed every 20 collections
        if self.successful_collections > 0 and self.successful_collections % 20 == 0:
            self.current_organism_speed_modifier += 0.05
        
        # Introduce new types every 5 collections
        if self.successful_collections > 0 and self.successful_collections % 5 == 0:
            self.organism_spawn_level = min(self.TOTAL_ORGANISM_TYPES, self.organism_spawn_level + 1)


    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["vel"] *= 0.95 # Drag

    def _cleanup_objects(self):
        self.projectiles = [p for p in self.projectiles if p["lifespan"] > 0]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _spawn_organism(self):
        radius = self.np_random.integers(6, 12)
        pos = pygame.Vector2(
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(radius, self.HEIGHT - radius)
        )
        type_id = self.np_random.integers(0, self.organism_spawn_level)
        color = self.ORGANISM_PALETTE[type_id]
        
        pattern_choice = self.np_random.choice(["linear", "circular"])
        
        organism = {"pos": pos, "radius": radius, "type_id": type_id, "color": color, "pattern": pattern_choice}

        if pattern_choice == "linear":
            speed = self.np_random.uniform(0.5, 1.5) * self.current_organism_speed_modifier
            angle = self.np_random.uniform(0, 2 * math.pi)
            organism["vel"] = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        elif pattern_choice == "circular":
            center = pos + pygame.Vector2(self.np_random.uniform(-50, 50), self.np_random.uniform(-50, 50))
            orbit_radius = pos.distance_to(center)
            orbit_speed = self.np_random.uniform(0.01, 0.03) * self.current_organism_speed_modifier
            organism["pattern_state"] = {
                "center": center, "radius": orbit_radius, "speed": orbit_speed, "angle": self.np_random.uniform(0, 2 * math.pi)
            }
        
        self.organisms.append(organism)

    def _get_dist_to_closest_uncollected(self):
        uncollected_orgs = [o for o in self.organisms if o["type_id"] not in self.collected_types]
        if not uncollected_orgs:
            return None
        
        closest_dist = float('inf')
        for o in uncollected_orgs:
            dist = self.player_pos.distance_to(o["pos"])
            if dist < closest_dist:
                closest_dist = dist
        return closest_dist

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if len(self.collected_types) == self.TOTAL_ORGANISM_TYPES:
            return True
        return False

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_organisms()
        self._render_trajectory_preview()
        self._render_player()
        self._render_projectiles()
        self._render_particles()

    def _create_background_elements(self):
        elements = []
        for _ in range(50):
            elements.append({
                "pos": (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                "radius": random.randint(5, 40),
                "color": (
                    self.COLOR_BG[0] + random.randint(5, 15),
                    self.COLOR_BG[1] + random.randint(5, 15),
                    self.COLOR_BG[2] + random.randint(5, 15),
                    random.randint(10, 30) # alpha
                )
            })
        return elements

    def _render_background(self):
        for el in self._background_elements:
            pygame.gfxdraw.filled_circle(self.screen, el["pos"][0], el["pos"][1], el["radius"], el["color"])

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow effect
        for i in range(4):
            radius = self.PLAYER_RADIUS + i * 3
            alpha = self.COLOR_PLAYER_GLOW[3] * (1 - i / 4)
            color = (self.COLOR_PLAYER_GLOW[0], self.COLOR_PLAYER_GLOW[1], self.COLOR_PLAYER_GLOW[2], int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_organisms(self):
        for o in self.organisms:
            pos = (int(o["pos"].x), int(o["pos"].y))
            color = o["color"]
            # Grey out collected types
            if o["type_id"] in self.collected_types:
                gray_val = int(sum(color) / 3)
                color = (gray_val, gray_val, gray_val)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], o["radius"], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], o["radius"], color)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p["radius"], self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p["radius"], self.COLOR_PROJECTILE)

    def _render_trajectory_preview(self):
        if not self.skill_tree_open:
            start_pos = self.player_pos + self.aim_direction * (self.PLAYER_RADIUS + 5)
            end_pos = start_pos + self.aim_direction * 75
            pygame.draw.line(self.screen, self.COLOR_TRAJECTORY, start_pos, end_pos, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / p["max_lifespan"]))
            color = (p["color"][0], p["color"][1], p["color"][2], int(alpha))
            size = int(p["radius"] * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                pos = (int(p["pos"].x), int(p["pos"].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, size), color)

    def _render_ui(self):
        # HUD
        progress = len(self.collected_types) / self.TOTAL_ORGANISM_TYPES
        progress_text = f"Survey: {progress:.0%}"
        score_text = f"Score: {self.score:.1f}"
        
        self._draw_text(progress_text, (10, 10), self.font_medium, self.COLOR_UI_TEXT)
        self._draw_text(score_text, (self.WIDTH - 10, 10), self.font_medium, self.COLOR_UI_TEXT, align="topright")
        
        # Skill points indicator
        skill_points_text = f"Skill Points: {self.skill_points}"
        self._draw_text(skill_points_text, (10, 40), self.font_small, self.COLOR_UI_TEXT)

        # Skill Tree Overlay
        if self.skill_tree_open:
            self._render_skill_tree()

    def _render_skill_tree(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 150)) # Darken background

        box_w, box_h = 400, 280
        box_x, box_y = (self.WIDTH - box_w) / 2, (self.HEIGHT - box_h) / 2
        pygame.draw.rect(s, self.COLOR_UI_BG, (box_x, box_y, box_w, box_h), border_radius=10)
        
        title_surf = self.font_large.render("SKILL TREE", True, self.COLOR_UI_TEXT)
        s.blit(title_surf, (box_x + (box_w - title_surf.get_width()) / 2, box_y + 20))

        start_y = box_y + 80
        for i, (key, skill) in enumerate(self.skills.items()):
            y_pos = start_y + i * 60
            
            # Highlight cursor
            if i == self.skill_cursor:
                pygame.draw.rect(s, (*self.COLOR_UI_HIGHLIGHT, 80), (box_x + 10, y_pos - 10, box_w - 20, 50), border_radius=5)
            
            # Skill name
            skill_name_surf = self.font_medium.render(skill["name"], True, self.COLOR_UI_TEXT)
            s.blit(skill_name_surf, (box_x + 20, y_pos))

            # Skill level bars
            level_text = f"Lvl: {skill['level']}/{skill['max_level']}"
            level_surf = self.font_small.render(level_text, True, self.COLOR_UI_TEXT)
            s.blit(level_surf, (box_x + 220, y_pos + 5))
            
            # Cost
            cost_text = f"Cost: {skill['cost']}"
            cost_color = self.COLOR_UI_HIGHLIGHT if self.skill_points >= skill['cost'] else (150, 150, 150)
            cost_surf = self.font_small.render(cost_text, True, cost_color)
            s.blit(cost_surf, (box_x + 320, y_pos + 5))

        self.screen.blit(s, (0, 0))

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)

    # --- Utility Methods ---
    def _create_particle_burst(self, position, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pygame.Vector2(position),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "radius": self.np_random.integers(1, 4),
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color
            })
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "survey_progress": len(self.collected_types) / self.TOTAL_ORGANISM_TYPES,
            "skill_points": self.skill_points
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The main loop is for human play and visualization, and will not run in a headless environment.
    # To see the game, you might need to unset the SDL_VIDEODRIVER variable.
    # For example:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Manual Play ---
    # This allows a human to play the game to test visuals and mechanics.
    try:
        pygame.display.set_caption("Microscopic Explorer")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        clock = pygame.time.Clock()
        running = True
        total_reward = 0

        while running:
            movement = 0 # no-op
            space = 0
            shift = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward:.2f}. Info: {info}")
                obs, info = env.reset()
                total_reward = 0
    except pygame.error as e:
        print("\nCould not create display. This is expected in a headless environment.")
        print("To play manually, you may need to run this script in a desktop environment.")
        print(f"Pygame error: {e}\n")
            
    env.close()