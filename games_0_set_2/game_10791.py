import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:04:29.171705
# Source Brief: brief_00791.md
# Brief Index: 791
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Terraform: A Rhythm/Platformer Gymnasium Environment

    The agent controls a character on a 2D plane, with the goal of terraforming a barren planet.
    This is achieved by successfully playing musical patterns.

    **Gameplay Loop:**
    1.  Observe scrolling musical notes. Each note corresponds to a specific vertical lane.
    2.  Move the character to the platform in the correct lane for the upcoming note.
    3.  Press the 'match' key (spacebar) when the note is in the target zone.
    4.  A successful match rewards the player, creates a new platform for the next note in the sequence,
        and contributes to the planet's terraforming progress.
    5.  Missing a note or falling off a platform results in failure.

    **Visuals:**
    - The game features a minimalist, vibrant aesthetic with glowing neon elements.
    - A starry background with a large, initially barren planet that gradually turns green.
    - The player, notes, and effects are rendered with smooth, anti-aliased shapes and particle effects.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]` (Movement): 0=None, 1=Up (Jump), 2=Down, 3=Left, 4=Right
    - `action[1]` (Match): 0=Released, 1=Pressed
    - `action[2]` (Unused): 0=Released, 1=Pressed

    **Observation Space:** `Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)`
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythm-platformer where you jump between platforms to match scrolling musical notes. "
        "Successful hits terraform a barren planet."
    )
    user_guide = "Controls: ←→ to move, ↑ to jump. Press space in the target zone to match the note."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TARGET_FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_STARS = (200, 200, 220)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_PLATFORM = (180, 180, 210)
    COLOR_PLATFORM_GLOW = (200, 200, 255, 30)
    COLOR_NOTE_UPCOMING = (0, 150, 255)
    COLOR_NOTE_SUCCESS = (0, 255, 100)
    COLOR_NOTE_MISS = (255, 50, 50)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PLANET_BARREN = (70, 60, 65)
    COLOR_PLANET_TERRAFORMED = (60, 180, 80)

    # Physics & Gameplay
    GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -14
    PLAYER_SPEED = 7
    NOTE_SPEED_INITIAL = 3.0
    NOTE_SPEED_INCREMENT = 0.05
    NOTE_TARGET_X = 120
    NOTE_TARGET_WIDTH = 40
    LANE_Y_POSITIONS = [300, 220, 140]

    # --- Initialization ---
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Musical Patterns: Each sub-list is a pattern. Numbers are lane indices (0-2).
        self.PATTERNS = [
            [0, 1, 0, 2, 1, 2, 0],
            [0, 0, 1, 1, 2, 2, 1, 0],
            [2, 1, 0, 1, 2, 0, 1, 2, 0],
            [0, 1, 1, 2, 2, 2, 1, 0, 0],
        ]
        self.total_notes_in_level = sum(len(p) for p in self.PATTERNS)
        
        # Initialize state variables to be defined in reset()
        self.player_pos = None
        self.player_vel = None
        self.on_ground = False
        self.camera_x = 0
        self.platforms = []
        self.notes = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_state = False
        self.current_pattern_index = 0
        self.current_note_in_pattern_index = 0
        self.note_speed = self.NOTE_SPEED_INITIAL
        self.successful_matches = 0
        self.np_random = None

    # --- Gymnasium Core Methods ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Player State
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False

        # World State
        self.camera_x = 0
        self.platforms = []
        self.notes = []
        self.particles = []
        self.stars = [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 3)) for _ in range(100)]
        
        # Gameplay State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_state = False
        self.current_pattern_index = 0
        self.current_note_in_pattern_index = 0
        self.note_speed = self.NOTE_SPEED_INITIAL
        self.successful_matches = 0

        # Initial Setup
        self._create_initial_platform()
        self._generate_notes_for_current_pattern()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack Action
        movement, space_held, _ = action
        space_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held

        # 2. Update Game Logic
        self._update_player(movement)
        self._update_world()
        
        # 3. Calculate Reward
        step_reward = self._handle_note_matching(space_pressed)

        # 4. Check Termination
        self.steps += 1
        terminated = False
        truncated = False
        if self.player_pos.y > self.SCREEN_HEIGHT + 50: # Fell off screen
            terminated = True
            step_reward = -100
        elif self.current_pattern_index >= len(self.PATTERNS): # Completed all patterns
            terminated = True
            step_reward = 100
        elif self.steps >= self.MAX_STEPS: # Max steps reached
            truncated = True
        
        if terminated or truncated:
            self.game_over = True

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    # --- State & Observation ---
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
            "successful_matches": self.successful_matches,
            "pattern": self.current_pattern_index,
        }

    # --- Game Logic Sub-routines ---
    def _update_player(self, movement):
        # Horizontal movement
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # Vertical movement (Jump)
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # Sound: Player Jump
            self._create_particles(self.player_pos + pygame.Vector2(0, 15), 5, (200, 200, 200), count=5, spread=5)

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, 15) # Terminal velocity

        # Update position
        self.player_pos += self.player_vel

        # Horizontal bounds
        self.player_pos.x = max(self.camera_x + 10, self.player_pos.x)
        self.player_pos.x = min(self.camera_x + self.SCREEN_WIDTH - 10, self.player_pos.x)
        
        # Collision with platforms
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel.y > 0:
                # Check if player was above the platform in the last frame
                if (self.player_pos.y - self.player_vel.y) <= plat.top:
                    self.player_pos.y = plat.top
                    self.player_vel.y = 0
                    self.on_ground = True
                    break

    def _update_world(self):
        # Smooth camera follow
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.08

        # Update notes
        for note in self.notes[:]:
            if note['pos'].x - self.camera_x < -50:
                self.notes.remove(note)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
    
    def _handle_note_matching(self, space_pressed):
        if not space_pressed:
            return 0

        reward = 0
        hit_a_note = False
        target_rect = pygame.Rect(self.NOTE_TARGET_X, 0, self.NOTE_TARGET_WIDTH, self.SCREEN_HEIGHT)

        for note in self.notes:
            note_screen_x = note['pos'].x - self.camera_x
            if target_rect.collidepoint(note_screen_x, note['pos'].y):
                player_lane = self._get_player_lane()
                if player_lane == note['lane']:
                    # --- SUCCESSFUL HIT ---
                    hit_a_note = True
                    note['hit'] = True
                    # Sound: Note Success
                    self._create_particles(note['pos'], 10, self.COLOR_NOTE_SUCCESS, count=20, spread=15)
                    
                    self.score += 1
                    self.successful_matches += 1
                    reward += 1

                    # Difficulty scaling
                    if self.successful_matches > 0 and self.successful_matches % 50 == 0:
                        self.note_speed += self.NOTE_SPEED_INCREMENT

                    # Advance pattern
                    self.current_note_in_pattern_index += 1
                    if self.current_note_in_pattern_index >= len(self.PATTERNS[self.current_pattern_index]):
                        # Pattern complete
                        self.current_pattern_index += 1
                        self.current_note_in_pattern_index = 0
                        self.score += 5
                        reward += 5
                        if self.current_pattern_index < len(self.PATTERNS):
                           self._generate_notes_for_current_pattern()
                    
                    # Generate next platform
                    self._generate_next_platform()
                    
                    # Remove the hit note after a short delay for visual feedback
                    note['lifetime'] = 5 
                    break 
        
        if not hit_a_note:
            # --- MISS ---
            # Sound: Note Miss
            self._create_particles(self.player_pos, 8, self.COLOR_NOTE_MISS, count=10, spread=10)
        
        return reward

    # --- Helper Methods ---
    def _create_initial_platform(self):
        initial_platform = pygame.Rect(self.player_pos.x - 100, self.LANE_Y_POSITIONS[0] + 20, 250, 20)
        self.platforms.append(initial_platform)
    
    def _generate_notes_for_current_pattern(self):
        pattern = self.PATTERNS[self.current_pattern_index]
        last_note_x = self.player_pos.x + 300
        for i, lane_index in enumerate(pattern):
            x_pos = last_note_x + (i + 1) * 200 + self.np_random.uniform(-20, 20)
            y_pos = self.LANE_Y_POSITIONS[lane_index]
            self.notes.append({
                'pos': pygame.Vector2(x_pos, y_pos),
                'lane': lane_index,
                'hit': False,
                'lifetime': -1 # Infinite until hit
            })

    def _generate_next_platform(self):
        if self.current_pattern_index >= len(self.PATTERNS):
            return # No more patterns

        pattern = self.PATTERNS[self.current_pattern_index]
        if self.current_note_in_pattern_index >= len(pattern):
            return # End of pattern

        # Find the note we need to build a platform for
        next_note_lane = pattern[self.current_note_in_pattern_index]
        
        # Position the platform relative to the player
        x_pos = self.player_pos.x + self.np_random.uniform(200, 300)
        y_pos = self.LANE_Y_POSITIONS[next_note_lane] + 20
        width = self.np_random.integers(120, 181)
        
        new_platform = pygame.Rect(x_pos, y_pos, width, 20)
        self.platforms.append(new_platform)

        # Prune old platforms
        while len(self.platforms) > 15:
            self.platforms.pop(0)

    def _get_player_lane(self):
        for i, lane_y in enumerate(self.LANE_Y_POSITIONS):
            if abs(self.player_pos.y - lane_y) < 40:
                return i
        return -1 # Not in any lane

    def _create_particles(self, pos, radius, color, count=10, spread=10):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy() + pygame.Vector2(self.np_random.uniform(-spread, spread), self.np_random.uniform(-spread, spread)),
                'vel': vel,
                'radius': self.np_random.uniform(radius/2, radius),
                'color': color,
                'lifetime': self.np_random.integers(15, 26)
            })

    # --- Rendering ---
    def _render_game(self):
        # Background Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STARS, (x, y, size, size))
            
        # Planet Terraforming
        planet_center = (self.SCREEN_WIDTH - 120, 120)
        planet_radius = 80
        progress_angle = 360 * (self.successful_matches / max(1, self.total_notes_in_level))
        pygame.gfxdraw.filled_circle(self.screen, planet_center[0], planet_center[1], planet_radius, self.COLOR_PLANET_BARREN)
        if progress_angle > 0:
            pygame.gfxdraw.pie(self.screen, planet_center[0], planet_center[1], planet_radius, -90, int(-90 + progress_angle), self.COLOR_PLANET_TERRAFORMED)
        pygame.gfxdraw.aacircle(self.screen, planet_center[0], planet_center[1], planet_radius, self.COLOR_PLATFORM)

        # Platforms
        for plat in self.platforms:
            screen_plat = plat.move(-self.camera_x, 0)
            glow_rect = screen_plat.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_PLATFORM_GLOW, s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_plat, border_radius=5)

        # Notes
        for note in self.notes:
            if note['lifetime'] > 0:
                note['lifetime'] -= 1
                if note['lifetime'] == 0:
                    self.notes.remove(note)
                    continue
            
            screen_pos = (int(note['pos'].x - self.camera_x), int(note['pos'].y))
            if screen_pos[0] < -20 or screen_pos[0] > self.SCREEN_WIDTH + 20:
                continue

            color = self.COLOR_NOTE_SUCCESS if note['hit'] else self.COLOR_NOTE_UPCOMING
            radius = 12
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius + 4, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, (255,255,255))

        # Particles
        for p in self.particles:
            screen_pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            alpha = int(255 * (p['lifetime'] / 25))
            color = (*p['color'], alpha)
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(p['radius']), color)

        # Player
        screen_pos = (int(self.player_pos.x - self.camera_x), int(self.player_pos.y))
        player_radius = 12
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], player_radius + 8, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], player_radius, (255, 255, 255))

    def _render_ui(self):
        # Target Zone
        target_zone_rect = pygame.Rect(self.NOTE_TARGET_X, 0, self.NOTE_TARGET_WIDTH, self.SCREEN_HEIGHT)
        s = pygame.Surface(target_zone_rect.size, pygame.SRCALPHA)
        s.fill((0, 255, 150, 20))
        self.screen.blit(s, target_zone_rect.topleft)
        pygame.draw.rect(self.screen, (0, 255, 150, 80), target_zone_rect, 1)

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Game Over / Win Text
        if self.game_over:
            if self.current_pattern_index >= len(self.PATTERNS):
                msg = "TERRAFORM COMPLETE"
                color = self.COLOR_NOTE_SUCCESS
            else:
                msg = "SEQUENCE LOST"
                color = self.COLOR_NOTE_MISS
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This block is for manual play and will not be run by the evaluation system.
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Terraform")
    clock = pygame.time.Clock()
    
    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1 # Jump
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3 # Left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4 # Right
            
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_pressed, shift_pressed]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            done = True

        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.TARGET_FPS)

    env.close()