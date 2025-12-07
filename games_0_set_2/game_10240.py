import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:03:25.124635
# Source Brief: brief_00240.md
# Brief Index: 240
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper function for linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t

# --- Helper Classes for Game Entities ---

class Cursor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pos = pygame.Vector2(width / 2, height / 2)
        self.speed = 8
        self.color = (255, 255, 255)
        self.size = 15

    def move(self, direction):
        if direction == 1:  # Up
            self.pos.y -= self.speed
        elif direction == 2:  # Down
            self.pos.y += self.speed
        elif direction == 3:  # Left
            self.pos.x -= self.speed
        elif direction == 4:  # Right
            self.pos.x += self.speed
        
        # Clamp position to screen bounds
        self.pos.x = np.clip(self.pos.x, self.size, self.width - self.size)
        self.pos.y = np.clip(self.pos.y, self.size, self.height - self.size)

    def draw(self, surface):
        x, y = int(self.pos.x), int(self.pos.y)
        pygame.draw.line(surface, self.color, (x - self.size, y), (x + self.size, y), 2)
        pygame.draw.line(surface, self.color, (x, y - self.size), (x, y + self.size), 2)
        pygame.gfxdraw.aacircle(surface, x, y, self.size, self.color)

class Player:
    def __init__(self, start_pos):
        self.pos = pygame.Vector2(start_pos)
        self.target_pos = pygame.Vector2(start_pos)
        self.color = (255, 255, 0)
        self.size = 10
        self.current_platform_idx = 0
        self.is_falling = False

    def update(self):
        if not self.is_falling:
            self.pos.x = lerp(self.pos.x, self.target_pos.x, 0.08)
            self.pos.y = lerp(self.pos.y, self.target_pos.y, 0.08)
        else:
            self.pos.y += 10 # Fall speed

    def set_target(self, new_target_pos, platform_idx):
        self.target_pos = pygame.Vector2(new_target_pos)
        self.current_platform_idx = platform_idx

    def draw(self, surface):
        x, y = int(self.pos.x), int(self.pos.y)
        points = [
            (x, y - self.size),
            (x - self.size // 2, y + self.size // 2),
            (x + self.size // 2, y + self.size // 2),
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)

class Note:
    def __init__(self, pos, note_type_info):
        self.pos = pygame.Vector2(pos)
        self.type_id = note_type_info['id']
        self.color = note_type_info['color']
        self.creation_time = pygame.time.get_ticks()
        self.lifetime = 1000 # ms
        self.pulse_size = 0
        self.max_pulse_size = 80

    def update(self):
        self.pulse_size = lerp(self.pulse_size, self.max_pulse_size, 0.1)
        return pygame.time.get_ticks() - self.creation_time < self.lifetime

    def draw(self, surface):
        x, y = int(self.pos.x), int(self.pos.y)
        # Draw the note core
        pygame.gfxdraw.aacircle(surface, x, y, 8, self.color)
        pygame.gfxdraw.filled_circle(surface, x, y, 8, self.color)
        # Draw the expanding pulse
        pulse_alpha = max(0, 255 * (1 - (self.pulse_size / self.max_pulse_size)))
        pulse_color = (*self.color, int(pulse_alpha))
        if self.pulse_size > 1:
            pygame.gfxdraw.aacircle(surface, x, y, int(self.pulse_size), pulse_color)

class Platform:
    def __init__(self, pos, size, melody, note_types):
        self.rect = pygame.Rect(pos, size)
        self.required_melody = set(melody)
        self.heard_notes_this_step = set()
        self.is_active = False
        self.activation_glow = 0.0
        self.note_types = note_types

    def hear_note(self, note_type_id):
        if note_type_id in self.required_melody:
            self.heard_notes_this_step.add(note_type_id)

    def check_activation(self):
        if not self.is_active and self.required_melody.issubset(self.heard_notes_this_step):
            self.is_active = True
            return True # Just activated
        return False

    def reset_heard(self):
        self.heard_notes_this_step.clear()

    def update(self):
        if self.is_active:
            self.activation_glow = lerp(self.activation_glow, 1.0, 0.1)
        else:
            self.activation_glow = lerp(self.activation_glow, 0.0, 0.1)

    def draw(self, surface):
        # Draw base
        pygame.draw.rect(surface, (60, 60, 80), self.rect, 0, 8)
        # Draw glow
        if self.activation_glow > 0.01:
            glow_color = (255, 255, 200)
            s = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            alpha = int(self.activation_glow * 150)
            pygame.draw.rect(s, (*glow_color, alpha), (0,0,*self.rect.size), border_radius=8)
            surface.blit(s, self.rect.topleft)
        # Draw outline
        pygame.draw.rect(surface, (150, 150, 180), self.rect, 2, 8)

        # Draw melody requirement icons
        icon_size = 10
        total_width = len(self.required_melody) * (icon_size + 2) - 2
        start_x = self.rect.centerx - total_width / 2
        for i, note_id in enumerate(sorted(list(self.required_melody))):
            color = self.note_types[note_id]['color']
            is_heard = note_id in self.heard_notes_this_step or self.is_active
            icon_color = color if is_heard else (80, 80, 80)
            center = (int(start_x + i * (icon_size + 2)), self.rect.centery)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], icon_size//2, icon_color)
            pygame.gfxdraw.aacircle(surface, center[0], center[1], icon_size//2, (200,200,200))


class Particle:
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        self.color = color
        self.lifetime = random.randint(20, 40)
        self.age = 0

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.age += 1
        return self.age < self.lifetime

    def draw(self, surface):
        alpha = 255 * (1 - self.age / self.lifetime)
        color = (*self.color, int(alpha))
        s = pygame.Surface((4, 4), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (2, 2), 2)
        surface.blit(s, (int(self.pos.x - 2), int(self.pos.y - 2)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Activate platforms by playing the correct sequence of notes to guide the player to the end of the level."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to play a note and shift to cycle through available notes."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.COLOR_BG_TOP = (10, 10, 20)
        self.COLOR_BG_BOTTOM = (30, 20, 40)

        self.NOTE_TYPES = [
            {'id': 0, 'name': 'C', 'color': (255, 80, 80)},
            {'id': 1, 'name': 'D', 'color': (80, 255, 80)},
            {'id': 2, 'name': 'E', 'color': (80, 120, 255)},
            {'id': 3, 'name': 'F', 'color': (255, 120, 255)},
            {'id': 4, 'name': 'G', 'color': (255, 255, 80)},
            {'id': 5, 'name': 'A', 'color': (80, 255, 255)},
        ]
        
        self.player = None
        self.cursor = None
        self.platforms = []
        self.notes = []
        self.particles = []
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        self.reset()
        # self.validate_implementation() # Commented out for final submission
    
    def _generate_level(self):
        self.platforms = []
        num_platforms = 15
        
        # First platform
        start_pos = (50, self.HEIGHT - 100)
        start_size = (100, 40)
        p = Platform(start_pos, start_size, [], self.NOTE_TYPES)
        p.is_active = True
        self.platforms.append(p)
        
        last_platform = p
        
        for i in range(1, num_platforms):
            dx = random.randint(80, 120)
            dy = random.randint(-60, 60)
            
            new_pos_x = last_platform.rect.right + dx
            new_pos_y = np.clip(last_platform.rect.centery + dy, 50, self.HEIGHT - 50)
            
            if new_pos_x > self.WIDTH - 150: # Wrap to next "screen"
                new_pos_x = 50
                new_pos_y = self.HEIGHT - 100 # Reset y to a predictable spot
            
            new_size = (random.randint(80, 150), 40)
            
            num_available_notes = min(len(self.NOTE_TYPES), 2 + i // 5)
            melody_length = min(num_available_notes, 1 + i // 10)
            
            melody = random.sample(range(num_available_notes), k=melody_length)
            
            new_platform = Platform((new_pos_x, new_pos_y - new_size[1]/2), new_size, melody, self.NOTE_TYPES)
            self.platforms.append(new_platform)
            last_platform = new_platform

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_level()
        
        self.player = Player(self.platforms[0].rect.center)
        self.cursor = Cursor(self.WIDTH, self.HEIGHT)
        
        self.notes = []
        self.particles = []
        
        self.available_note_types = 2
        self.selected_note_idx = 0
        
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        self.cursor.move(movement)

        if shift_pressed:
            # SFX: UI_Cycle.wav
            self.selected_note_idx = (self.selected_note_idx + 1) % self.available_note_types
        
        for p in self.platforms:
            p.reset_heard()

        if space_pressed:
            # SFX: Note_Clone.wav
            note_info = self.NOTE_TYPES[self.selected_note_idx]
            self.notes.append(Note(self.cursor.pos, note_info))
            for _ in range(30):
                self.particles.append(Particle(self.cursor.pos, note_info['color']))
        
        # Update and check notes/pulses
        self.notes = [n for n in self.notes if n.update()]
        for note in self.notes:
            for p in self.platforms:
                dist = pygame.Vector2(p.rect.center).distance_to(note.pos)
                if dist < note.pulse_size + max(p.rect.width, p.rect.height)/2:
                    p.hear_note(note.type_id)

        # Update platforms
        for i, p in enumerate(self.platforms):
            p.update()
            if p.check_activation():
                # SFX: Platform_Activate.wav
                reward += 1.0
                self.score += 1
                self.available_note_types = min(len(self.NOTE_TYPES), 2 + self.score // 5)
                # Give reward for notes that contributed
                reward += 0.1 * len(p.required_melody)

        # Update player and check for progression
        self.player.update()
        if not self.player.is_falling:
            current_platform_idx = self.player.current_platform_idx
            next_platform_idx = current_platform_idx + 1

            if next_platform_idx >= len(self.platforms):
                # On the final platform, check for arrival to end the game
                dist_to_target = self.player.pos.distance_to(self.player.target_pos)
                if dist_to_target < 5 and not self.game_over:
                    reward += 100
                    terminated = True
                    self.game_over = True
            else:
                # Not at the end, check if we can move to the next platform
                current_p = self.platforms[current_platform_idx]
                next_p = self.platforms[next_platform_idx]

                # Player is at rest if its target is the current platform's center and it has arrived
                is_at_rest = (self.player.target_pos == pygame.Vector2(current_p.rect.center) and
                              self.player.pos.distance_to(self.player.target_pos) < 5)

                if is_at_rest and next_p.is_active:
                    # Next platform is active, so move to it
                    self.player.set_target(next_p.rect.center, next_platform_idx)
                # If not at rest, or if next platform is not active, the player continues its current action (moving or waiting)

        if self.player.pos.y > self.HEIGHT + 20 and not self.game_over:
            reward -= 100
            terminated = True
            self.game_over = True

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        self.steps += 1
        truncated = False
        if self.steps >= 2000:
            truncated = True
            terminated = True # For compatibility, often term and trunc are both set
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            color = lerp(self.COLOR_BG_TOP[0], self.COLOR_BG_BOTTOM[0], y / self.HEIGHT), \
                    lerp(self.COLOR_BG_TOP[1], self.COLOR_BG_BOTTOM[1], y / self.HEIGHT), \
                    lerp(self.COLOR_BG_TOP[2], self.COLOR_BG_BOTTOM[2], y / self.HEIGHT)
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        for p in self.platforms:
            p.draw(self.screen)
        
        for particle in self.particles:
            particle.draw(self.screen)
            
        for note in self.notes:
            note.draw(self.screen)
            
        self.player.draw(self.screen)
        self.cursor.draw(self.screen)

    def _render_ui(self):
        # Display selected note
        selected_note_info = self.NOTE_TYPES[self.selected_note_idx]
        note_text = self.font.render(f"NOTE: {selected_note_info['name']}", True, (255, 255, 255))
        self.screen.blit(note_text, (10, 10))
        pygame.draw.rect(self.screen, selected_note_info['color'], (120, 15, 15, 15))

        # Display progress
        progress_text = self.font.render(f"PROGRESS: {self.score}/{len(self.platforms)-1}", True, (255, 255, 255))
        self.screen.blit(progress_text, (self.WIDTH - progress_text.get_width() - 10, 10))

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_platform": self.player.current_platform_idx,
            "available_notes": self.available_note_types
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Use a display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Puzzle Platformer")
    
    while running:
        if terminated or truncated:
            obs, info = env.reset()
            terminated = False
            truncated = False

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # None
        if keys[pygame.K_w] or keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()