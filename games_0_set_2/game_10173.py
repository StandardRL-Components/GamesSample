import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:56:38.861522
# Source Brief: brief_00173.md
# Brief Index: 173
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a visually-rich, physics-based game.
    The player controls a glowing orb in a haunted space station, flipping gravity,
    magnetizing components for repair, and matching rhythmic beats to survive.
    """
    metadata = {"render_modes": ["rgb_array"]}

    auto_advance = True
    game_description = (
        "Control a glowing orb in a haunted space station. Flip gravity, magnetize components for repair, and match rhythmic beats to survive."
    )
    user_guide = (
        "Controls: ↑/↓ to flip gravity, ←/→ to move. Hold space to magnetize components. Press shift on the beat to repel ghosts."
    )

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 15, 26)
    COLOR_PLAYER = (0, 170, 255)
    COLOR_WALL = (48, 64, 80)
    COLOR_COMPONENT = (255, 255, 0)
    COLOR_GHOST = (255, 0, 68)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_SYSTEM_OK = (0, 255, 128)
    COLOR_SYSTEM_DMG = (255, 128, 0)
    COLOR_BEAT_INDICATOR = (255, 255, 255)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Physics
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.92
    GRAVITY_ACCEL = 0.35
    COMPONENT_MAGNET_FORCE = 0.5
    COMPONENT_FRICTION = 0.95
    MAX_VEL = 8.0

    # Game Parameters
    MAX_STEPS = 2000
    NUM_SYSTEMS = 3
    COMPONENTS_PER_SYSTEM = 3
    NUM_GHOSTS = 2
    PLAYER_RADIUS = 12
    COMPONENT_RADIUS = 6
    GHOST_RADIUS = 15
    MAGNET_RADIUS = 120
    INITIAL_BEAT_INTERVAL = 60  # in steps (2 seconds at 30fps)
    BEAT_WINDOW = 4 # +/- steps from the beat

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.gravity_direction = 1  # 1 for down, -1 for up
        self.components = []
        self.ghosts = []
        self.walls = []
        self.systems = []
        self.particles = []
        self.beat_timer = 0
        self.beat_interval = self.INITIAL_BEAT_INTERVAL
        self.beat_hit_this_cycle = False
        self.last_shift_state = 0
        self.ghost_speed_multiplier = 1.0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity_direction = 1
        self.beat_interval = self.INITIAL_BEAT_INTERVAL
        self.beat_timer = self.beat_interval
        self.beat_hit_this_cycle = False
        self.ghost_speed_multiplier = 1.0
        self.last_shift_state = 0
        
        self._procedurally_generate_level()

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)

        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input and State Changes ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game Logic ---
        self._update_player_physics()
        self._update_components(space_held)
        self._update_ghosts()
        self._update_particles()
        reward += self._update_beat_mechanic(shift_held)
        
        # --- Handle Collisions and Interactions ---
        reward += self._handle_collisions()
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.ghost_speed_multiplier += 0.05
        if self.steps > 0 and self.steps % 500 == 0:
            self.beat_interval = max(20, self.beat_interval * 0.9) # Increase frequency, cap at 0.66s

        # --- Termination Conditions ---
        terminated = self.game_over
        repaired_count = sum(1 for s in self.systems if s['repaired'])
        if repaired_count == self.NUM_SYSTEMS:
            reward += 100
            terminated = True
            # Sound: Win Jingle
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        truncated = False # This environment does not truncate based on time limit.
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Gravity Flip
        if movement == 1 and self.gravity_direction == 1: # Up
            self.gravity_direction = -1
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 2.0)
            # Sound: Gravity Shift Up
        elif movement == 2 and self.gravity_direction == -1: # Down
            self.gravity_direction = 1
            self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 2.0)
            # Sound: Gravity Shift Down

        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        if movement == 4: # Right
            self.player_vel[0] += self.PLAYER_ACCEL

    def _update_player_physics(self):
        # Apply gravity
        self.player_vel[1] += self.gravity_direction * self.GRAVITY_ACCEL
        
        # Apply friction
        self.player_vel[0] *= self.PLAYER_FRICTION
        
        # Clamp velocity
        self.player_vel = np.clip(self.player_vel, -self.MAX_VEL, self.MAX_VEL)

        # Update position
        self.player_pos += self.player_vel

        # Wall collisions
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        for wall in self.walls:
            if player_rect.colliderect(wall):
                # Determine collision side and correct position/velocity
                if self.player_vel[0] > 0 and player_rect.right > wall.left and player_rect.left < wall.left:
                    self.player_pos[0] = wall.left - self.PLAYER_RADIUS
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0 and player_rect.left < wall.right and player_rect.right > wall.right:
                    self.player_pos[0] = wall.right + self.PLAYER_RADIUS
                    self.player_vel[0] = 0
                
                if self.player_vel[1] > 0 and player_rect.bottom > wall.top and player_rect.top < wall.top:
                    self.player_pos[1] = wall.top - self.PLAYER_RADIUS
                    self.player_vel[1] = 0
                elif self.player_vel[1] < 0 and player_rect.top < wall.bottom and player_rect.bottom > wall.bottom:
                    self.player_pos[1] = wall.bottom + self.PLAYER_RADIUS
                    self.player_vel[1] = 0

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)


    def _update_components(self, space_held):
        for comp in self.components:
            if space_held:
                dist_vec = self.player_pos - comp['pos']
                dist = np.linalg.norm(dist_vec)
                if dist > 0 and dist < self.MAGNET_RADIUS:
                    force = dist_vec / dist * self.COMPONENT_MAGNET_FORCE * (1 - dist/self.MAGNET_RADIUS)
                    comp['vel'] += force
                    comp['magnetized'] = True
                else:
                    comp['magnetized'] = False
            else:
                comp['magnetized'] = False

            comp['vel'] *= self.COMPONENT_FRICTION
            comp['pos'] += comp['vel']
            
            # Wall collisions for components
            comp_rect = pygame.Rect(comp['pos'][0] - self.COMPONENT_RADIUS, comp['pos'][1] - self.COMPONENT_RADIUS, self.COMPONENT_RADIUS*2, self.COMPONENT_RADIUS*2)
            for wall in self.walls:
                if comp_rect.colliderect(wall):
                    # A simple stop is sufficient for components
                    if comp['vel'][0] > 0: comp['pos'][0] = wall.left - self.COMPONENT_RADIUS
                    elif comp['vel'][0] < 0: comp['pos'][0] = wall.right + self.COMPONENT_RADIUS
                    if comp['vel'][1] > 0: comp['pos'][1] = wall.top - self.COMPONENT_RADIUS
                    elif comp['vel'][1] < 0: comp['pos'][1] = wall.bottom + self.COMPONENT_RADIUS
                    comp['vel'] *= 0 # Stop completely on wall hit

            comp['pos'][0] = np.clip(comp['pos'][0], self.COMPONENT_RADIUS, self.SCREEN_WIDTH - self.COMPONENT_RADIUS)
            comp['pos'][1] = np.clip(comp['pos'][1], self.COMPONENT_RADIUS, self.SCREEN_HEIGHT - self.COMPONENT_RADIUS)

    def _update_ghosts(self):
        for ghost in self.ghosts:
            t = (self.steps + ghost['offset']) * 0.02 * self.ghost_speed_multiplier
            if ghost['path_type'] == 'sin_h':
                ghost['pos'][0] = ghost['origin'][0] + math.sin(t) * ghost['amplitude']
            elif ghost['path_type'] == 'sin_v':
                ghost['pos'][1] = ghost['origin'][1] + math.sin(t) * ghost['amplitude']
            elif ghost['path_type'] == 'circle':
                ghost['pos'][0] = ghost['origin'][0] + math.cos(t) * ghost['amplitude']
                ghost['pos'][1] = ghost['origin'][1] + math.sin(t) * ghost['amplitude']

    def _update_beat_mechanic(self, shift_held):
        reward = 0
        self.beat_timer -= 1
        
        is_in_window = abs(self.beat_timer) < self.BEAT_WINDOW or abs(self.beat_interval - self.beat_timer) < self.BEAT_WINDOW

        # Detect rising edge of shift press
        shift_pressed_now = shift_held and not self.last_shift_state
        self.last_shift_state = shift_held

        if shift_pressed_now and is_in_window and not self.beat_hit_this_cycle:
            self.beat_hit_this_cycle = True
            # Sound: Beat Success
            self._create_particles(self.player_pos, 30, self.COLOR_BEAT_INDICATOR, 4.0, 40)
            # Repel nearby ghosts
            for ghost in self.ghosts:
                dist_vec = ghost['pos'] - self.player_pos
                dist = np.linalg.norm(dist_vec)
                if dist < 150 and dist > 0:
                    ghost['origin'] += dist_vec / dist * 20 # Push their origin point away

        if self.beat_timer <= 0:
            if not self.beat_hit_this_cycle:
                reward -= 0.1 # Missed beat penalty
                # Sound: Beat Fail
                self.score -= 0.1
            self.beat_timer = self.beat_interval
            self.beat_hit_this_cycle = False
        
        return reward

    def _handle_collisions(self):
        reward = 0
        # Player-Component
        collected_indices = []
        for i, comp in enumerate(self.components):
            dist = np.linalg.norm(self.player_pos - comp['pos'])
            if dist < self.PLAYER_RADIUS + self.COMPONENT_RADIUS:
                collected_indices.append(i)
                reward += 0.1
                self.score += 0.1
                # Sound: Collect Component
                self._create_particles(comp['pos'], 15, self.COLOR_COMPONENT, 2.5)
                
                # Find a system that needs this component
                for system in self.systems:
                    if not system['repaired'] and system['collected'] < self.COMPONENTS_PER_SYSTEM:
                        system['collected'] += 1
                        if system['collected'] == self.COMPONENTS_PER_SYSTEM:
                            system['repaired'] = True
                            reward += 1.0
                            self.score += 1.0
                            # Sound: System Repaired
                            self._create_particles(system['pos'], 50, self.COLOR_SYSTEM_OK, 5.0, 60)
                        break

        # Remove collected components
        for i in sorted(collected_indices, reverse=True):
            del self.components[i]

        # Player-Ghost
        for ghost in self.ghosts:
            dist = np.linalg.norm(self.player_pos - ghost['pos'])
            if dist < self.PLAYER_RADIUS + self.GHOST_RADIUS:
                self.game_over = True
                reward -= 10
                # Sound: Player Death
                self._create_particles(self.player_pos, 100, self.COLOR_GHOST, 6.0, 50)
                break
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        repaired_count = sum(1 for s in self.systems if s['repaired'])
        return {
            "score": self.score,
            "steps": self.steps,
            "systems_repaired": repaired_count,
            "total_systems": self.NUM_SYSTEMS,
            "gravity": "up" if self.gravity_direction == -1 else "down",
        }
        
    def _render_game(self):
        # Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Ghosts
        for ghost in self.ghosts:
            self._draw_swirl(ghost['pos'], self.GHOST_RADIUS, self.COLOR_GHOST)

        # Components
        for comp in self.components:
            pos = tuple(map(int, comp['pos']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.COMPONENT_RADIUS, self.COLOR_COMPONENT)
            if comp['magnetized']:
                pulse_alpha = 100 + math.sin(self.steps * 0.3) * 50
                self._draw_glow(pos, self.COMPONENT_RADIUS * 1.5, self.COLOR_COMPONENT, int(pulse_alpha))

        # Player
        player_pos_int = tuple(map(int, self.player_pos))
        self._draw_glow(player_pos_int, self.PLAYER_RADIUS * 2.5, self.COLOR_PLAYER, 60)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # Particles
        for p in self.particles:
            p_pos = tuple(map(int, p['pos']))
            alpha = p['color'][3] * (p['life'] / p['max_life'])
            color = (*p['color'][:3], int(alpha))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, p_pos, size)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # System Status
        for i, system in enumerate(self.systems):
            status_pos = (self.SCREEN_WIDTH - 30 - i * 25, 15)
            color = self.COLOR_SYSTEM_OK if system['repaired'] else self.COLOR_SYSTEM_DMG
            if not system['repaired'] and self.steps % 10 < 5: # Flicker if damaged
                color = tuple(c//2 for c in color)
            pygame.draw.rect(self.screen, color, (*status_pos, 20, 20))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (*status_pos, 20, 20), 1)

        # Gravity Indicator
        arrow_y = self.player_pos[1]
        arrow_x = 20
        if self.gravity_direction == 1: # Down
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(arrow_x, arrow_y+8), (arrow_x-8, arrow_y-8), (arrow_x+8, arrow_y-8)])
        else: # Up
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(arrow_x, arrow_y-8), (arrow_x-8, arrow_y+8), (arrow_x+8, arrow_y+8)])

        # Beat Indicator
        beat_progress = self.beat_timer / self.beat_interval
        indicator_pos = (self.SCREEN_WIDTH // 2, 30)
        is_in_window = abs(self.beat_timer) < self.BEAT_WINDOW or abs(self.beat_interval - self.beat_timer) < self.BEAT_WINDOW
        
        if is_in_window:
            flash_radius = 15 + math.sin(self.steps * 0.5) * 5
            self._draw_glow(indicator_pos, flash_radius, self.COLOR_BEAT_INDICATOR, 150)
        
        pygame.draw.circle(self.screen, self.COLOR_WALL, indicator_pos, 12)
        arc_rect = pygame.Rect(indicator_pos[0]-12, indicator_pos[1]-12, 24, 24)
        pygame.draw.arc(self.screen, self.COLOR_BEAT_INDICATOR, arc_rect, math.pi/2, math.pi/2 + (1-beat_progress) * 2 * math.pi, 3)

        if self.game_over:
            repaired_count = sum(1 for s in self.systems if s['repaired'])
            msg = "SYSTEMS REPAIRED" if repaired_count == self.NUM_SYSTEMS else "CONNECTION LOST"
            msg_text = self.font_msg.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _render_background_effects(self):
        # Static stars
        if self.np_random:
            for _ in range(50):
                x = (self.np_random.integers(0, self.SCREEN_WIDTH) + self.steps // 20) % self.SCREEN_WIDTH
                y = self.np_random.integers(0, self.SCREEN_HEIGHT)
                alpha = self.np_random.integers(30, 80)
                pygame.draw.circle(self.screen, (*self.COLOR_WALL, alpha), (x, y), 1)

    def _draw_glow(self, pos, radius, color, alpha):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), radius)
        self.screen.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_swirl(self, pos, radius, color):
        num_blobs = 5
        for i in range(num_blobs):
            angle = self.steps * 0.05 + (i * 2 * math.pi / num_blobs)
            offset_radius = radius * 0.5
            blob_pos = (
                int(pos[0] + math.cos(angle) * offset_radius),
                int(pos[1] + math.sin(angle) * offset_radius)
            )
            blob_radius = int(radius * 0.6 + math.sin(angle * 2) * radius * 0.1)
            self._draw_glow(blob_pos, blob_radius, color, 20)

    def _create_particles(self, pos, count, color, speed, lifespan=30):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.5, 1.0) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': lifespan,
                'max_life': lifespan,
                'color': (*color, 255),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1

    def _procedurally_generate_level(self):
        self.walls.clear()
        self.components.clear()
        self.ghosts.clear()
        self.systems.clear()

        # Create outer boundary walls
        self.walls.append(pygame.Rect(0, 0, self.SCREEN_WIDTH, 5))
        self.walls.append(pygame.Rect(0, self.SCREEN_HEIGHT-5, self.SCREEN_WIDTH, 5))
        self.walls.append(pygame.Rect(0, 0, 5, self.SCREEN_HEIGHT))
        self.walls.append(pygame.Rect(self.SCREEN_WIDTH-5, 0, 5, self.SCREEN_HEIGHT))
        
        # Create some internal platforms
        for _ in range(4):
            w = self.np_random.integers(100, 300)
            h = 10
            x = self.np_random.integers(20, self.SCREEN_WIDTH - w - 20)
            y = self.np_random.integers(50, self.SCREEN_HEIGHT - h - 50)
            self.walls.append(pygame.Rect(x, y, w, h))
        
        # Create systems
        for i in range(self.NUM_SYSTEMS):
            self.systems.append({
                'id': i,
                'repaired': False,
                'collected': 0,
                'pos': (self.SCREEN_WIDTH - 40 - i * 25, 25) # UI position, not physical
            })
        
        # Create components
        num_components = self.NUM_SYSTEMS * self.COMPONENTS_PER_SYSTEM
        for _ in range(num_components):
            self.components.append({
                'pos': self._get_random_valid_pos(self.COMPONENT_RADIUS),
                'vel': np.zeros(2, dtype=np.float32),
                'magnetized': False
            })

        # Create ghosts
        for _ in range(self.NUM_GHOSTS):
            origin = self._get_random_valid_pos(self.GHOST_RADIUS * 2)
            self.ghosts.append({
                'origin': origin,
                'pos': origin.copy(),
                'path_type': self.np_random.choice(['sin_h', 'sin_v', 'circle']),
                'amplitude': self.np_random.integers(50, 150),
                'offset': self.np_random.integers(0, 100)
            })

    def _get_random_valid_pos(self, radius):
        while True:
            pos = np.array([
                self.np_random.uniform(radius, self.SCREEN_WIDTH - radius),
                self.np_random.uniform(radius, self.SCREEN_HEIGHT - radius)
            ])
            rect = pygame.Rect(pos[0]-radius, pos[1]-radius, radius*2, radius*2)
            
            # Check for overlap with walls or center of screen
            is_valid = True
            if rect.collidelist(self.walls) != -1:
                is_valid = False
            if np.linalg.norm(pos - np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])) < 100:
                is_valid = False # Don't spawn on top of player's start

            if is_valid:
                return pos

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For this to work, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Space Station Repair")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # ARROWS: Move/Flip Gravity
    # SPACE: Magnetize
    # SHIFT: Beat Match
    
    while not done:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        # Pygame Event Handling for quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if done:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Display the observation from the environment ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    env.close()