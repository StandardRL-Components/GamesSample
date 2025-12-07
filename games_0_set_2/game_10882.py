import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:13:06.145813
# Source Brief: brief_00882.md
# Brief Index: 882
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates dual gravity wells
    to guide falling orbs into synchronized target zones.

    **Visuals:**
    - Minimalist, geometric style with a dark space-themed background.
    - Orbs glow with colors corresponding to their point value.
    - Gravity wells have a radial field effect that visualizes their strength.
    - Particle effects provide satisfying feedback for scoring events.

    **Gameplay:**
    - Control the strength of two static gravity wells (left and right).
    - Orbs spawn from the top and are pulled by the wells.
    - Guide orbs into target zones to score points.
    - A significant score bonus is awarded for synchronizing multiple orbs
      into the same target within a short time frame.
    - The game ends by reaching the target score (win), letting an orb
      go off-screen (loss), or reaching the step limit.

    **Action Space `MultiDiscrete([5, 2, 2])`:**
    - `action[0]` (Movement):
        - 0: No-op
        - 1 (Up): Increase left well strength
        - 2 (Down): Decrease left well strength
        - 3 (Left): Decrease right well strength
        - 4 (Right): Increase right well strength
    - `action[1]` (Space): Unused
    - `action[2]` (Shift): Unused
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Manipulate dual gravity wells to guide falling orbs into target zones. Score bonuses by synchronizing multiple orbs into the same target."
    user_guide = "Controls: ↑/↓ to adjust the left gravity well's strength, ←/→ to adjust the right gravity well's strength."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_SCORE = 300
    MAX_STEPS = 5000
    FPS = 30  # Assumed frame rate for physics simulation

    # Colors
    COLOR_BG_START = (10, 0, 20)
    COLOR_BG_END = (30, 0, 50)
    COLOR_WELL = (255, 255, 255)
    COLOR_TARGET = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    ORB_DEFINITIONS = {
        "green": {"color": (50, 255, 150), "value": 10},
        "blue": {"color": (100, 150, 255), "value": 20},
        "purple": {"color": (200, 100, 255), "value": 30},
    }

    # Physics & Gameplay
    WELL_STRENGTH_MIN = 1
    WELL_STRENGTH_MAX = 10
    WELL_FORCE_MULTIPLIER = 80.0
    ORB_DRAG = 0.99
    ORB_BASE_GRAVITY = 0.05
    SYNC_WINDOW_FRAMES = 30  # 1 second at 30 FPS

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        self.bg_surface = self._create_background()

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.well_left_strength = 0
        self.well_right_strength = 0
        self.well_left_pos = (0, 0)
        self.well_right_pos = (0, 0)
        self.orbs = []
        self.targets = []
        self.particles = []
        self.sync_events = deque()
        self.sync_count = 0
        self.orb_spawn_timer = 0
        self.orb_spawn_rate = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Wells
        self.well_left_strength = 5
        self.well_right_strength = 5
        self.well_left_pos = (self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.25)
        self.well_right_pos = (self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT * 0.25)

        # Entities
        self.orbs = []
        self.particles = []
        self.sync_events = deque()
        self.sync_count = 0

        # Timers & Difficulty
        self.orb_spawn_timer = 0
        self.orb_spawn_rate = 1.0 / (self.FPS * 2) # 0.5 orbs per second

        self._generate_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # --- 1. Handle Action ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        if movement == 1: # Up: Increase left well
            self.well_left_strength = min(self.WELL_STRENGTH_MAX, self.well_left_strength + 1)
        elif movement == 2: # Down: Decrease left well
            self.well_left_strength = max(self.WELL_STRENGTH_MIN, self.well_left_strength - 1)
        elif movement == 3: # Left: Decrease right well
            self.well_right_strength = max(self.WELL_STRENGTH_MIN, self.well_right_strength - 1)
        elif movement == 4: # Right: Increase right well
            self.well_right_strength = min(self.WELL_STRENGTH_MAX, self.well_right_strength + 1)

        # --- 2. Update Game Logic ---
        self.steps += 1
        reward = 0

        # Update difficulty
        self.orb_spawn_rate = (1.0 / (self.FPS * 2)) + (self.score // 100) * 0.01

        self._update_spawner()
        reward += self._update_orbs()
        reward += self._update_sync_events()
        self._update_particles()

        # --- 3. Calculate Termination & Final Reward ---
        terminated = False
        truncated = False
        if self.score >= self.TARGET_SCORE:
            terminated = True
            self.game_over = True
            reward += 100
            # Sound: Win
        elif self.game_over: # Loss condition (orb out of bounds)
            terminated = True
            reward -= 100
            # Sound: Lose
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        if self.sync_count > 0 and self.sync_count % 10 == 0:
            self._generate_targets()
            self.sync_count = 0 # Prevent re-triggering

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods: Game Logic ---

    def _update_spawner(self):
        self.orb_spawn_timer += self.orb_spawn_rate
        if self.orb_spawn_timer >= 1.0:
            self.orb_spawn_timer -= 1.0
            orb_type = random.choice(list(self.ORB_DEFINITIONS.keys()))
            self.orbs.append({
                "pos": np.array([random.uniform(100, self.SCREEN_WIDTH - 100), -20.0]),
                "vel": np.array([random.uniform(-0.5, 0.5), 1.0]),
                "radius": 10,
                "type": orb_type,
                "color": self.ORB_DEFINITIONS[orb_type]["color"],
                "value": self.ORB_DEFINITIONS[orb_type]["value"],
                "last_dist_to_target": float('inf')
            })
            # Sound: Spawn

    def _update_orbs(self):
        reward = 0
        for i in range(len(self.orbs) - 1, -1, -1):
            orb = self.orbs[i]
            
            # --- Physics ---
            # Well forces
            force_l = self._calculate_force(orb['pos'], self.well_left_pos, self.well_left_strength)
            force_r = self._calculate_force(orb['pos'], self.well_right_pos, self.well_right_strength)
            total_force = force_l + force_r
            
            # Gravity and Drag
            total_force[1] += self.ORB_BASE_GRAVITY
            orb['vel'] += total_force
            orb['vel'] *= self.ORB_DRAG
            orb['pos'] += orb['vel']

            # --- Continuous Reward ---
            min_dist = float('inf')
            for target in self.targets:
                dist = np.linalg.norm(orb['pos'] - target['pos'])
                if dist < min_dist:
                    min_dist = dist
            
            if min_dist < orb['last_dist_to_target']:
                reward += 0.01
            else:
                reward -= 0.01
            orb['last_dist_to_target'] = min_dist

            # --- Collision & Scoring ---
            for target_idx, target in enumerate(self.targets):
                if np.linalg.norm(orb['pos'] - target['pos']) < target['radius'] + orb['radius']:
                    self.score += orb['value']
                    reward += 1
                    self.sync_events.append({
                        "frame": self.steps,
                        "target_id": target_idx,
                        "value": orb['value']
                    })
                    self._create_particles(target['pos'], orb['color'], 30)
                    del self.orbs[i]
                    # Sound: Score
                    break
            else:
                # --- Boundary Check ---
                if not (0 < orb['pos'][0] < self.SCREEN_WIDTH and orb['pos'][1] < self.SCREEN_HEIGHT + 50):
                    self.game_over = True
                    del self.orbs[i]
        return reward

    def _update_sync_events(self):
        reward = 0
        # Remove old events
        while self.sync_events and self.steps - self.sync_events[0]["frame"] > self.SYNC_WINDOW_FRAMES:
            self.sync_events.popleft()

        # Check for syncs
        if len(self.sync_events) > 1:
            events_by_target = {}
            for event in self.sync_events:
                tid = event['target_id']
                if tid not in events_by_target:
                    events_by_target[tid] = []
                events_by_target[tid].append(event)
            
            processed_indices = set()
            for tid, events in events_by_target.items():
                if len(events) > 1:
                    sync_value = sum(e['value'] for e in events)
                    bonus = sync_value * 2
                    self.score += bonus
                    reward += bonus
                    self.sync_count += 1
                    # Sound: Sync Bonus
                    self._create_particles(self.targets[tid]['pos'], self.COLOR_TEXT, 50, speed=5)
                    for i in range(len(self.sync_events) - 1, -1, -1):
                        if self.sync_events[i]['target_id'] == tid:
                            del self.sync_events[i]
        return reward

    def _calculate_force(self, pos1, pos2, strength):
        diff = pos2 - pos1
        dist_sq = np.dot(diff, diff)
        if dist_sq < 1: dist_sq = 1 # Prevent division by zero and extreme forces
        
        force_magnitude = self.WELL_FORCE_MULTIPLIER * strength / dist_sq
        force_vec = diff / np.sqrt(dist_sq) * force_magnitude
        return force_vec
    
    def _generate_targets(self):
        self.targets.clear()
        # Sound: New Targets
        num_targets = self.np_random.integers(2, 4)
        y_pos = self.SCREEN_HEIGHT * 0.85
        
        x_positions = np.linspace(self.SCREEN_WIDTH * 0.15, self.SCREEN_WIDTH * 0.85, num_targets)
        for x in x_positions:
            self.targets.append({
                "pos": np.array([x, y_pos]),
                "radius": 25
            })
            
    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_wells()
        self._render_targets()
        self._render_particles()
        self._render_orbs()

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            alpha = y / self.SCREEN_HEIGHT
            color = [
                int((1 - alpha) * s + alpha * e)
                for s, e in zip(self.COLOR_BG_START, self.COLOR_BG_END)
            ]
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_wells(self):
        self._draw_well(self.well_left_pos, self.well_left_strength)
        self._draw_well(self.well_right_pos, self.well_right_strength)

    def _draw_well(self, pos, strength):
        x, y = int(pos[0]), int(pos[1])
        # Draw gradient field effect
        max_radius = 15 + strength * 8
        for i in range(strength):
            alpha = 1 - (i / strength)
            radius = int(max_radius * alpha)
            color = (self.COLOR_WELL[0], self.COLOR_WELL[1], self.COLOR_WELL[2], int(30 * alpha))
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        # Draw core
        pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_WELL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_WELL)
    
    def _render_targets(self):
        for target in self.targets:
            x, y = int(target['pos'][0]), int(target['pos'][1])
            r = target['radius']
            color_dim = tuple(c // 2 for c in self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, color_dim)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, x, y, r-2, self.COLOR_TARGET)

    def _render_orbs(self):
        for orb in self.orbs:
            self._draw_glowing_circle(
                self.screen,
                orb['pos'],
                orb['radius'],
                orb['color']
            )

    def _draw_glowing_circle(self, surface, pos, radius, color):
        x, y = int(pos[0]), int(pos[1])
        glow_radius = int(radius * 1.8)
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(surface, x, y, glow_radius, glow_color)
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius*1.4), (*color, 80))
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)

    def _create_particles(self, pos, color, count, speed=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, 1.0) * speed
            vel = np.array([math.cos(angle) * vel_mag, math.sin(angle) * vel_mag])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": random.randint(15, 30),
                "color": color
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                del self.particles[i]
    
    def _render_particles(self):
        for p in self.particles:
            alpha = p['lifetime'] / 30.0
            size = int(max(1, 5 * alpha))
            color = (*p['color'], int(255 * alpha))
            pygame.draw.circle(self.screen, color, p['pos'].astype(int), size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Well Strengths
        left_text = self.font_small.render(f"{self.well_left_strength}", True, self.COLOR_TEXT)
        right_text = self.font_small.render(f"{self.well_right_strength}", True, self.COLOR_TEXT)
        self.screen.blit(left_text, (int(self.well_left_pos[0] - 5), int(self.well_left_pos[1] - 30)))
        self.screen.blit(right_text, (int(self.well_right_pos[0] - 5), int(self.well_right_pos[1] - 30)))

    # --- Gymnasium Interface Methods ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "orbs_on_screen": len(self.orbs),
            "well_strengths": (self.well_left_strength, self.well_right_strength)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Well Synchronizer")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("W/S: Adjust Left Well Strength")
    print("A/D: Adjust Right Well Strength")
    print("Q: Quit")

    action = [0, 0, 0] # Start with a no-op action
    
    while not done:
        # Action is based on key holds, not events, for continuous control
        keys = pygame.key.get_pressed()
        
        # Reset movement action
        movement_action = 0
        if keys[pygame.K_w]:
            movement_action = 1 # Up
        elif keys[pygame.K_s]:
            movement_action = 2 # Down
        
        # Use A/D for right well control for more intuitive keyboard layout
        # Note: This overwrites W/S if both are pressed.
        # A more robust system might handle combined inputs if the action space allowed.
        if keys[pygame.K_a]:
            movement_action = 3 # Left
        elif keys[pygame.K_d]:
            movement_action = 4 # Right

        action[0] = movement_action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            done = True

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()