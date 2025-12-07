import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls a chain reaction of teleporting
    quantum particles to guide them to target zones within a time limit.

    **Visuals:** Abstract, minimalist style with glowing particles and effects.
    **Gameplay:** Select a particle and a teleport direction. Teleporting triggers
    a chain reaction in nearby particles. A new particle is added for each one
    that reaches its target, increasing complexity.
    **Objective:** Get all particles into their designated target zones before the
    timer runs out.
    """
    game_description = "Guide quantum particles to their target zones by triggering teleportation chain reactions before time runs out."
    user_guide = "Use ↑/↓ to select a particle and ←/→ to aim. Press space to teleport the selected particle and trigger a chain reaction."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Gameplay parameters
    INITIAL_PARTICLES = 3
    TELEPORT_DISTANCE = 50
    CHAIN_REACTION_RADIUS = 60
    PARTICLE_RADIUS = 8
    TARGET_ZONE_RADIUS = 25

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_SELECTION = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_BAR = (0, 160, 255)
    TARGET_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (50, 205, 50),   # Lime Green
        (255, 20, 147),  # Deep Pink
        (30, 144, 255),  # Dodger Blue
        (255, 69, 0),    # Orange Red
        (123, 104, 238), # Medium Slate Blue
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles = []
        self.selected_particle_idx = 0
        self.selected_direction_idx = 0
        self.visual_effects = []
        self.last_space_held = False
        self.directions = self._create_directions()

    def _create_directions(self):
        """Creates a list of 8 direction vectors."""
        return [
            pygame.Vector2(0, -1),   # N
            pygame.Vector2(1, -1).normalize(),  # NE
            pygame.Vector2(1, 0),    # E
            pygame.Vector2(1, 1).normalize(),   # SE
            pygame.Vector2(0, 1),    # S
            pygame.Vector2(-1, 1).normalize(),  # SW
            pygame.Vector2(-1, 0),   # W
            pygame.Vector2(-1, -1).normalize(), # NW
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.visual_effects = []
        self.last_space_held = False

        self._generate_level(self.INITIAL_PARTICLES)
        self.selected_particle_idx = 0
        self.selected_direction_idx = 0

        return self._get_observation(), self._get_info()

    def _generate_level(self, num_particles):
        """Creates a new set of particles and their target zones."""
        self.particles = []
        
        # Use np_random for sampling to ensure seeding works
        color_indices = self.np_random.choice(len(self.TARGET_COLORS), num_particles, replace=False)
        used_colors = [self.TARGET_COLORS[i] for i in color_indices]
        
        padding = 50
        for i in range(num_particles):
            particle = {
                'pos': pygame.Vector2(
                    self.np_random.uniform(padding, self.SCREEN_WIDTH - padding),
                    self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
                ),
                'target_pos': pygame.Vector2(
                    self.np_random.uniform(padding, self.SCREEN_WIDTH - padding),
                    self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
                ),
                'target_color': used_colors[i],
                'in_zone': False,
                'id': i
            }
            self.particles.append(particle)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- Handle Actions ---
        reward += self._handle_actions(action)
        
        # --- Update Game State ---
        self._update_visual_effects()
        self._check_particle_states()
        
        terminated = self._check_termination()
        if terminated:
            # Apply terminal rewards
            if all(p['in_zone'] for p in self.particles):
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        """Processes the agent's action and returns immediate rewards."""
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Action 0: no-op
        # Action 1 & 2: Cycle particle selection
        if movement == 1: # Up -> Next particle
            if self.particles: self.selected_particle_idx = (self.selected_particle_idx + 1) % len(self.particles)
        elif movement == 2: # Down -> Previous particle
            if self.particles: self.selected_particle_idx = (self.selected_particle_idx - 1) % len(self.particles)
        
        # Action 3 & 4: Cycle teleport direction
        if movement == 3: # Left -> CCW
             self.selected_direction_idx = (self.selected_direction_idx - 1 + 8) % 8
        elif movement == 4: # Right -> CW
             self.selected_direction_idx = (self.selected_direction_idx + 1) % 8

        # Action: Space button (teleport)
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.particles:
            # Teleport sound placeholder: # sfx_teleport.play()
            reward += self._teleport_and_chain_reaction()

        self.last_space_held = space_held
        return reward

    def _teleport_and_chain_reaction(self):
        """Performs the teleport and triggers chain reactions, returning rewards."""
        reward = 0
        
        # Store pre-teleport distances to calculate distance-based reward
        pre_teleport_dist_sum = sum(p['pos'].distance_to(p['target_pos']) for p in self.particles)

        primary_particle = self.particles[self.selected_particle_idx]
        direction_vec = self.directions[self.selected_direction_idx]
        
        triggered_particles = [primary_particle]
        for other_p in self.particles:
            if other_p['id'] != primary_particle['id']:
                if primary_particle['pos'].distance_to(other_p['pos']) <= self.CHAIN_REACTION_RADIUS:
                    triggered_particles.append(other_p)
        
        # Add visual effect for the chain reaction
        self.visual_effects.append({
            'type': 'ring', 'pos': primary_particle['pos'], 'radius': 0, 
            'max_radius': self.CHAIN_REACTION_RADIUS, 'life': 15, 'max_life': 15,
            'color': self.COLOR_SELECTION
        })

        for p in triggered_particles:
            start_pos = pygame.Vector2(p['pos']) # FIX: Use constructor to copy
            p['pos'] += direction_vec * self.TELEPORT_DISTANCE
            
            # Screen wrapping
            p['pos'].x %= self.SCREEN_WIDTH
            p['pos'].y %= self.SCREEN_HEIGHT
            
            # Add teleport trail effect
            self.visual_effects.append({
                'type': 'trail', 'start': start_pos, 'end': p['pos'], 'life': 20, 
                'max_life': 20, 'color': p['target_color']
            })

        # Calculate distance-based reward
        post_teleport_dist_sum = sum(p['pos'].distance_to(p['target_pos']) for p in self.particles)
        distance_change = pre_teleport_dist_sum - post_teleport_dist_sum
        reward += distance_change * 0.05 # Scaled reward for moving closer/further

        return reward

    def _check_particle_states(self):
        """Check if particles are in their zones and add new particles if needed."""
        newly_completed = 0
        for p in self.particles:
            was_in_zone = p['in_zone']
            is_in_zone = p['pos'].distance_to(p['target_pos']) <= self.TARGET_ZONE_RADIUS
            p['in_zone'] = is_in_zone
            if is_in_zone and not was_in_zone:
                # Event-based reward for entering a zone
                self.score += 10.0 
                newly_completed += 1
                # Success sound placeholder: # sfx_success.play()

        if newly_completed > 0 and len(self.particles) < len(self.TARGET_COLORS):
             self._add_new_particle()

    def _add_new_particle(self):
        """Adds one new particle and target to the game."""
        current_colors = {p['target_color'] for p in self.particles}
        available_colors = [c for c in self.TARGET_COLORS if c not in current_colors]
        if not available_colors:
            return

        padding = 50
        new_particle = {
            'pos': pygame.Vector2(
                self.np_random.uniform(padding, self.SCREEN_WIDTH - padding),
                self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
            ),
            'target_pos': pygame.Vector2(
                self.np_random.uniform(padding, self.SCREEN_WIDTH - padding),
                self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
            ),
            'target_color': available_colors[0],
            'in_zone': False,
            'id': max(p['id'] for p in self.particles) + 1 if self.particles else 0
        }
        self.particles.append(new_particle)
        # New particle sound placeholder: # sfx_spawn.play()

    def _update_visual_effects(self):
        """Updates lifetimes and properties of all active visual effects."""
        for effect in self.visual_effects[:]:
            effect['life'] -= 1
            if effect['type'] == 'ring':
                effect['radius'] = effect['max_radius'] * (1 - effect['life'] / effect['max_life'])
            if effect['life'] <= 0:
                self.visual_effects.remove(effect)

    def _check_termination(self):
        """Checks for win/loss conditions."""
        time_out = self.steps >= self.MAX_STEPS
        all_in_zones = all(p['in_zone'] for p in self.particles) if self.particles else False

        if time_out or all_in_zones:
            self.game_over = True
            return True
        return False
    
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
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "particles_in_zone": sum(1 for p in self.particles if p['in_zone']),
            "total_particles": len(self.particles)
        }
        
    def _render_game(self):
        """Renders all primary game elements."""
        # Render target zones
        for p in self.particles:
            # Draw a filled circle for the target zone
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['target_pos'].x), int(p['target_pos'].y),
                self.TARGET_ZONE_RADIUS, (*p['target_color'], 50)
            )
            # Draw an anti-aliased outline
            pygame.gfxdraw.aacircle(
                self.screen, int(p['target_pos'].x), int(p['target_pos'].y),
                self.TARGET_ZONE_RADIUS, p['target_color']
            )

        # Render visual effects
        for effect in self.visual_effects:
            if effect['type'] == 'trail':
                alpha = int(255 * (effect['life'] / effect['max_life']))
                color = (*effect['color'], alpha)
                pygame.draw.line(self.screen, color, effect['start'], effect['end'], 2)
            elif effect['type'] == 'ring':
                alpha = int(150 * (effect['life'] / effect['max_life']))
                color = (*effect['color'], alpha)
                pygame.gfxdraw.aacircle(self.screen, int(effect['pos'].x), int(effect['pos'].y), int(effect['radius']), color)

        # Render particles
        for i, p in enumerate(self.particles):
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            # Glow effect
            glow_radius = int(self.PARTICLE_RADIUS * 2.5)
            glow_alpha = 60 if i == self.selected_particle_idx else 40
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, glow_radius, (*self.COLOR_PARTICLE, glow_alpha))
            # Core particle
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, self.PARTICLE_RADIUS, self.COLOR_PARTICLE)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, self.PARTICLE_RADIUS, self.COLOR_PARTICLE)

        # Render selection and direction indicator
        if self.particles:
            p = self.particles[self.selected_particle_idx]
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            
            # Pulsating selection ring
            pulse = abs(math.sin(self.steps * 0.2))
            sel_radius = self.PARTICLE_RADIUS + 5 + int(pulse * 3)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, sel_radius, self.COLOR_SELECTION)

            # Direction indicator
            direction_vec = self.directions[self.selected_direction_idx]
            start_point = p['pos'] + direction_vec * (sel_radius + 2)
            end_point = start_point + direction_vec * 15
            pygame.draw.line(self.screen, self.COLOR_SELECTION, start_point, end_point, 2)


    def _render_ui(self):
        """Renders the UI overlay."""
        # Score / Particle Count
        in_zone_count = sum(1 for p in self.particles if p['in_zone'])
        total_particles = len(self.particles)
        score_text = f"COMPLETED: {in_zone_count}/{total_particles}  |  SCORE: {self.score:.1f}"
        score_surface = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surface, (10, 10))

        # Timer bar
        time_ratio = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
        time_ratio = max(0, time_ratio)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, bar_y, int(bar_width * time_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Game Over Text
        if self.game_over:
            all_in_zones = all(p['in_zone'] for p in self.particles) if self.particles else False
            if all_in_zones:
                end_text = "SUCCESS"
                end_color = (100, 255, 100)
            else:
                end_text = "TIME UP"
                end_color = (255, 100, 100)

            end_surface = pygame.font.SysFont("Consolas", 60, bold=True).render(end_text, True, end_color)
            text_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_surface, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    manual_mode = True 
    
    # Create a display for manual play
    pygame.display.set_caption("Quantum Chain Reaction")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    total_reward = 0
    
    action = [0, 0, 0] # [movement, space, shift]

    while running:
        if manual_mode:
            # --- Manual Control ---
            action = [0, 0, 0] # Reset action each frame
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
                
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("--- RESET ---")
                    obs, info = env.reset()
                    total_reward = 0

        else:
            # --- Agent Control ---
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Info:", info)
            if manual_mode:
                # In manual mode, wait for reset
                pass
            else:
                obs, info = env.reset()
                total_reward = 0

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(GameEnv.FPS)

    env.close()