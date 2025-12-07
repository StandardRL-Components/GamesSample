import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:52:16.160523
# Source Brief: brief_00721.md
# Brief Index: 721
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player manipulates gravity wells to guide
    falling orbs into matching colored slots.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide falling orbs into their matching colored slots by manipulating the strength of powerful gravity wells."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to strengthen a corresponding gravity well. "
        "Hold Shift to strengthen all wells, or Space to weaken all wells."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 58)
    COLOR_UI_TEXT = (220, 220, 240)
    ORB_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]
    COLOR_WELL = (255, 255, 255)

    # Physics & Game Parameters
    ORB_RADIUS = 10
    SLOT_RADIUS = 14
    MAX_ORBS = 4
    GRAVITY_CONSTANT = 2000
    MIN_WELL_STRENGTH = 0.1
    MAX_WELL_STRENGTH = 2.0
    WELL_STRENGTH_DEFAULT = 0.5
    COLLISION_DAMPING = 0.95

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 16)

        # --- State Initialization ---
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.orbs = []
        self.slots = []
        self.wells = []
        self.walls = []
        self.particles = []
        self.strength_increment = 0.1
        self.last_orb_distances = {}

        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.strength_increment = 0.1
        self.particles.clear()

        # --- Initialize Game Elements ---
        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # 1. Handle player actions
        self._handle_input(action)

        # 2. Update game logic
        self._update_physics()
        self._handle_collisions()
        reward += self._check_slotting()
        self._update_particles()

        # 3. Calculate rewards
        reward += self._calculate_reward()

        # 4. Check for termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if not self.orbs and not truncated: # Win condition
                reward += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_level(self):
        """Initializes orbs, slots, wells, and walls for the current level."""
        self.orbs.clear()
        self.slots.clear()
        self.wells.clear()
        self.walls.clear()

        # Create slots at the bottom
        num_entities = min(self.MAX_ORBS, len(self.ORB_COLORS))
        slot_spacing = self.WIDTH / (num_entities + 1)
        for i in range(num_entities):
            pos_x = slot_spacing * (i + 1)
            self.slots.append({
                "pos": np.array([pos_x, self.HEIGHT - 30.0]),
                "color": self.ORB_COLORS[i]
            })

        # Create orbs at the top
        for i in range(num_entities):
            pos_x = self.np_random.uniform(50, self.WIDTH - 50)
            pos_y = self.np_random.uniform(50, 100)
            self.orbs.append({
                "pos": np.array([pos_x, pos_y]),
                "vel": np.array([0.0, 0.0]),
                "color": self.ORB_COLORS[i],
                "id": i
            })

        # Create gravity wells
        self.wells = [
            {"pos": np.array([self.WIDTH * 0.5, self.HEIGHT * 0.25]), "strength": self.WELL_STRENGTH_DEFAULT},
            {"pos": np.array([self.WIDTH * 0.5, self.HEIGHT * 0.75]), "strength": self.WELL_STRENGTH_DEFAULT},
            {"pos": np.array([self.WIDTH * 0.25, self.HEIGHT * 0.5]), "strength": self.WELL_STRENGTH_DEFAULT},
            {"pos": np.array([self.WIDTH * 0.75, self.HEIGHT * 0.5]), "strength": self.WELL_STRENGTH_DEFAULT},
        ]
        
        # Add walls based on level
        if self.level > 1:
            wall_y = 200 + (self.level - 2) * 20
            wall_width = 150 + (self.level - 2) * 10
            self.walls.append(pygame.Rect(
                self.WIDTH/2 - wall_width/2, wall_y, wall_width, 10
            ))
        
        # Reset distance tracking for reward calculation
        self.last_orb_distances.clear()
        for orb in self.orbs:
            slot_pos = self.slots[orb["id"]]["pos"]
            self.last_orb_distances[orb["id"]] = np.linalg.norm(orb["pos"] - slot_pos)


    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Arrow keys: Increase strength of a specific well
        if 1 <= movement <= 4:
            self.wells[movement - 1]["strength"] += self.strength_increment

        # Space: Decrease strength of all wells
        if space_held:
            for well in self.wells:
                well["strength"] -= self.strength_increment / 2 # Slower decrease

        # Shift: Increase strength of all wells
        if shift_held:
            for well in self.wells:
                well["strength"] += self.strength_increment / 2 # Slower increase

        # Clamp well strengths
        for well in self.wells:
            well["strength"] = np.clip(well["strength"], self.MIN_WELL_STRENGTH, self.MAX_WELL_STRENGTH)

    def _update_physics(self):
        for orb in self.orbs:
            total_force = np.array([0.0, 0.0])
            for well in self.wells:
                vec = well["pos"] - orb["pos"]
                dist_sq = np.dot(vec, vec)
                if dist_sq < 1: dist_sq = 1 # Avoid division by zero
                
                # Force = G * strength / dist^2
                force_magnitude = self.GRAVITY_CONSTANT * well["strength"] / dist_sq
                force_dir = vec / np.sqrt(dist_sq)
                total_force += force_dir * force_magnitude
            
            # Update velocity and position (F=ma, assume mass=1, so a=F)
            orb["vel"] += total_force / self.FPS
            orb["pos"] += orb["vel"] / self.FPS

    def _handle_collisions(self):
        # Orb-wall collisions
        for orb in self.orbs:
            if orb["pos"][0] < self.ORB_RADIUS:
                orb["pos"][0] = self.ORB_RADIUS
                orb["vel"][0] *= -self.COLLISION_DAMPING
            elif orb["pos"][0] > self.WIDTH - self.ORB_RADIUS:
                orb["pos"][0] = self.WIDTH - self.ORB_RADIUS
                orb["vel"][0] *= -self.COLLISION_DAMPING
            
            if orb["pos"][1] < self.ORB_RADIUS:
                orb["pos"][1] = self.ORB_RADIUS
                orb["vel"][1] *= -self.COLLISION_DAMPING
            elif orb["pos"][1] > self.HEIGHT - self.ORB_RADIUS:
                orb["pos"][1] = self.HEIGHT - self.ORB_RADIUS
                orb["vel"][1] *= -self.COLLISION_DAMPING

        # Orb-orb collisions
        for i in range(len(self.orbs)):
            for j in range(i + 1, len(self.orbs)):
                orb1, orb2 = self.orbs[i], self.orbs[j]
                vec = orb1["pos"] - orb2["pos"]
                dist = np.linalg.norm(vec)
                if dist < 2 * self.ORB_RADIUS:
                    # Resolve overlap
                    overlap = (2 * self.ORB_RADIUS - dist) / 2
                    orb1["pos"] += (vec / dist) * overlap
                    orb2["pos"] -= (vec / dist) * overlap
                    
                    # Elastic collision
                    v1, v2 = orb1["vel"], orb2["vel"]
                    x1, x2 = orb1["pos"], orb2["pos"]
                    # # SFX: orb_collide.wav
                    orb1["vel"] = v1 - (np.dot(v1-v2, x1-x2) / np.linalg.norm(x1-x2)**2) * (x1-x2) * self.COLLISION_DAMPING
                    orb2["vel"] = v2 - (np.dot(v2-v1, x2-x1) / np.linalg.norm(x2-x1)**2) * (x2-x1) * self.COLLISION_DAMPING
                    self._create_particles( (orb1['pos'] + orb2['pos'])/2, (180, 180, 180), 5, 2.0)

        # Orb-static wall collisions
        for orb in self.orbs:
            for wall in self.walls:
                if wall.collidepoint(orb['pos']):
                    # A simple resolution: push out and reflect velocity
                    # This is not physically perfect but works for simple axis-aligned rects
                    if wall.top < orb['pos'][1] < wall.bottom:
                        if orb['vel'][0] > 0: orb['pos'][0] = wall.left - self.ORB_RADIUS
                        else: orb['pos'][0] = wall.right + self.ORB_RADIUS
                        orb['vel'][0] *= -self.COLLISION_DAMPING
                    if wall.left < orb['pos'][0] < wall.right:
                        if orb['vel'][1] > 0: orb['pos'][1] = wall.top - self.ORB_RADIUS
                        else: orb['pos'][1] = wall.bottom + self.ORB_RADIUS
                        orb['vel'][1] *= -self.COLLISION_DAMPING


    def _check_slotting(self):
        reward = 0
        orbs_to_remove = []
        for i, orb in enumerate(self.orbs):
            # Each orb has a designated slot based on color/id
            slot = self.slots[orb["id"]]
            if orb["color"] == slot["color"]:
                dist = np.linalg.norm(orb["pos"] - slot["pos"])
                if dist < self.SLOT_RADIUS:
                    # # SFX: orb_slot.wav
                    orbs_to_remove.append(i)
                    self.score += 10
                    reward += 1.0 # Event-based reward
                    self._create_particles(slot["pos"], orb["color"], 50, 4.0)
        
        if orbs_to_remove:
            for i in sorted(orbs_to_remove, reverse=True):
                del self.orbs[i]
        
        # Level up if all orbs are slotted
        if not self.orbs and not self.game_over:
            self.score += 50 # Level clear bonus
            self.level += 1
            self.strength_increment *= 1.25 # Increase difficulty
            self._setup_level()

        return reward

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for getting closer to the correct slot
        current_distances = {}
        for orb in self.orbs:
            slot_pos = self.slots[orb["id"]]["pos"]
            dist = np.linalg.norm(orb["pos"] - slot_pos)
            current_distances[orb["id"]] = dist
            
            if orb["id"] in self.last_orb_distances:
                if dist < self.last_orb_distances[orb["id"]]:
                    reward += 0.1 # Approaching reward
            
        self.last_orb_distances = current_distances

        # Small penalty for high well strength to encourage efficiency
        total_strength = sum(well["strength"] for well in self.wells)
        reward -= 0.001 * total_strength

        return reward

    def _check_termination(self):
        # Terminate if all orbs are gone (win)
        return not self.orbs and self.level > self.MAX_ORBS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_slots()
        self._render_wells()
        self._render_walls()
        self._render_orbs()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---
    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_slots(self):
        for slot in self.slots:
            pos = (int(slot["pos"][0]), int(slot["pos"][1]))
            # Draw a slightly larger, darker background for the slot
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLOT_RADIUS, tuple(c//2 for c in slot["color"]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLOT_RADIUS, tuple(c//2 for c in slot["color"]))
            # Draw the inner, brighter part
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLOT_RADIUS - 3, slot["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLOT_RADIUS - 3, slot["color"])


    def _render_wells(self):
        for well in self.wells:
            pos = (int(well["pos"][0]), int(well["pos"][1]))
            strength_ratio = (well["strength"] - self.MIN_WELL_STRENGTH) / (self.MAX_WELL_STRENGTH - self.MIN_WELL_STRENGTH)
            
            # Pulsating effect
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
            
            # Outer glow
            glow_radius = int(20 + 30 * strength_ratio)
            glow_alpha = int(30 + 50 * strength_ratio * pulse)
            glow_color = self.COLOR_WELL + (glow_alpha,)
            
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Core circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_WELL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_WELL)

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_GRID, wall)

    def _render_orbs(self):
        for orb in self.orbs:
            pos = (int(orb["pos"][0]), int(orb["pos"][1]))
            color = orb["color"]
            
            # Glow effect
            glow_radius = self.ORB_RADIUS + 4
            glow_color = color + (80,) # Add alpha
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Orb body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_RADIUS, color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        level_text = self.font_level.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": color
            })

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p["pos"] += p["vel"] * 0.2
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1
            if p["life"] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / 40.0
            radius = int(life_ratio * 4)
            if radius > 0:
                alpha = int(life_ratio * 255)
                color = p["color"] + (alpha,)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Loop ---
    # Un-comment the line below to run with a display window
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gravity Well")
    clock = pygame.time.Clock()

    total_reward = 0
    
    # Map keyboard keys to MultiDiscrete actions
    # The wells are at: 0.5/0.25 (top), 0.5/0.75 (bottom), 0.25/0.5 (left), 0.75/0.5 (right)
    # Mapping keys to these positions logically
    key_map = {
        pygame.K_UP: 1,      # Top well
        pygame.K_DOWN: 2,    # Bottom well
        pygame.K_LEFT: 3,    # Left well
        pygame.K_RIGHT: 4,   # Right well
    }

    running = True
    while running:
        # --- Action Mapping for Human ---
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset on completion
            obs, info = env.reset()
            total_reward = 0


        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()