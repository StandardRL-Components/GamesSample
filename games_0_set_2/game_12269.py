import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:28:06.935830
# Source Brief: brief_02269.md
# Brief Index: 2269
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Symphony of the Scrapyard: A Gymnasium environment where an agent must
    magnetize falling musical notes and teleport them to the correct platforms
    to compose a melody, terraforming a desolate landscape in the process.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Capture falling musical notes with a magnetic beam and teleport them to the correct platforms "
        "to compose a melody and terraform a desolate scrapyard."
    )
    user_guide = (
        "Controls: Use arrow keys (←↑→) to aim the beam. Hold Shift to capture a note. "
        "Press Space to teleport the captured note to the targeted platform."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.TARGET_MELODY = ['C', 'D', 'E', 'F', 'G']

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors and visual constants
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_SCRAP = (40, 30, 50)
        self.COLOR_BEAM = (0, 255, 255)
        self.COLOR_BEAM_GLOW = (0, 150, 150)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)
        self.NOTE_MAP = {
            'C': {'color': (255, 80, 80), 'symbol': 'C'},
            'D': {'color': (255, 165, 0), 'symbol': 'D'},
            'E': {'color': (255, 255, 0), 'symbol': 'E'},
            'F': {'color': (0, 255, 0), 'symbol': 'F'},
            'G': {'color': (0, 150, 255), 'symbol': 'G'},
        }

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_beam_angle = 0.0
        self.player_beam_target_angle = 0.0
        self.magnet_center = (0, 0)
        self.falling_notes = []
        self.captured_note = None
        self.target_platforms = []
        self.placed_notes = []
        self.melody_progress = 0
        self.note_spawn_timer = 0
        self.note_fall_speed = 0.0
        self.successful_notes_count = 0
        self.particles = []
        self.terraform_level = 0.0
        self.last_space_held = False
        self.scrapyard_pieces = []
        
        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.melody_progress = 0
        self.terraform_level = 0.0
        
        self.magnet_center = (self.WIDTH // 2, self.HEIGHT - 20)
        self.player_beam_angle = -math.pi / 2
        self.player_beam_target_angle = -math.pi / 2
        
        self.falling_notes = []
        self.captured_note = None
        self.placed_notes = []
        self.particles = []
        
        self.note_fall_speed = 1.0
        self.successful_notes_count = 0
        self.note_spawn_timer = 0
        self.last_space_held = False

        # Generate static scrapyard background
        self.scrapyard_pieces = []
        for _ in range(20):
            w = self.np_random.integers(50, 150)
            h = self.np_random.integers(10, 40)
            x = self.np_random.integers(0, self.WIDTH - w)
            y = self.np_random.integers(0, self.HEIGHT - h)
            self.scrapyard_pieces.append(pygame.Rect(x, y, w, h))

        # Setup target platforms
        self.target_platforms = []
        num_platforms = len(self.TARGET_MELODY)
        for i, note_name in enumerate(self.TARGET_MELODY):
            platform_x = self.WIDTH * (i + 1) / (num_platforms + 1)
            platform_y = self.HEIGHT * 0.75
            self.target_platforms.append({
                'pos': (platform_x, platform_y),
                'note_name': note_name,
                'radius': 25
            })
        
        self._spawn_note()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- 1. Handle Input ---
        if movement == 1: # Up
            self.player_beam_target_angle = -math.pi / 2
        elif movement == 2: # Down (alias for Up, as down is not useful)
             self.player_beam_target_angle = -math.pi / 2
        elif movement == 3: # Left
            self.player_beam_target_angle = -math.pi * 0.9
        elif movement == 4: # Right
            self.player_beam_target_angle = -math.pi * 0.1

        # --- 2. Update Game State ---
        self.player_beam_angle += (self.player_beam_target_angle - self.player_beam_angle) * 0.2
        
        for note in self.falling_notes[:]:
            note['pos'][1] += self.note_fall_speed
            if note['pos'][1] > self.HEIGHT + note['radius']:
                self.falling_notes.remove(note)
                if self.melody_progress < len(self.TARGET_MELODY) and note['name'] == self.TARGET_MELODY[self.melody_progress]:
                    self.game_over = True
                    reward -= 5.0
                    self._create_particles(note['pos'], self.COLOR_FAIL, 50, 5)
                    # sfx: fail_sound

        if self.captured_note:
            beam_end_x = self.magnet_center[0] + math.cos(self.player_beam_angle) * 150
            beam_end_y = self.magnet_center[1] + math.sin(self.player_beam_angle) * 150
            self.captured_note['pos'] = [beam_end_x, beam_end_y]
            
            if not shift_held:
                self.falling_notes.append(self.captured_note)
                self.captured_note = None
            else:
                reward -= 0.1 # Penalty for holding

        self._update_particles()
        
        target_terraform = self.melody_progress / len(self.TARGET_MELODY)
        self.terraform_level += (target_terraform - self.terraform_level) * 0.05
        
        # --- 3. Handle Interactions ---
        space_pressed = space_held and not self.last_space_held
        
        if space_pressed and self.captured_note:
            closest_platform = min(self.target_platforms, 
                                   key=lambda p: abs(math.atan2(p['pos'][1] - self.magnet_center[1], p['pos'][0] - self.magnet_center[0]) - self.player_beam_angle))
            
            if self.melody_progress < len(self.TARGET_MELODY):
                required_note_name = self.TARGET_MELODY[self.melody_progress]
                if self.captured_note['name'] == required_note_name and closest_platform['note_name'] == required_note_name:
                    reward += 5.0
                    self.score += 10
                    self.placed_notes.append({'pos': closest_platform['pos'], 'name': self.captured_note['name'], 'radius': 20})
                    self.melody_progress += 1
                    self.successful_notes_count += 1
                    self._create_particles(closest_platform['pos'], self.COLOR_SUCCESS, 50, 3)
                    # sfx: success_chime
                    
                    if self.successful_notes_count > 0 and self.successful_notes_count % 5 == 0:
                         self.note_fall_speed += 0.05
                else:
                    reward -= 5.0
                    self.score -= 5
                    self.game_over = True
                    self._create_particles(self.captured_note['pos'], self.COLOR_FAIL, 100, 5)
                    # sfx: error_buzz
            
            self.captured_note = None

        elif not self.captured_note and shift_held:
            beam_end_x = self.magnet_center[0] + math.cos(self.player_beam_angle) * 150
            beam_end_y = self.magnet_center[1] + math.sin(self.player_beam_angle) * 150
            beam_end = np.array([beam_end_x, beam_end_y])

            for note in self.falling_notes[:]:
                note_pos = np.array(note['pos'])
                if np.linalg.norm(beam_end - note_pos) < note['radius'] + 15:
                    self.captured_note = note
                    self.falling_notes.remove(note)
                    reward += 1.0
                    self.score += 1
                    # sfx: capture_zap
                    break

        # --- 4. Handle Game Flow ---
        self.note_spawn_timer -= 1
        if self.note_spawn_timer <= 0 and self.melody_progress < len(self.TARGET_MELODY):
            if not any(n['name'] == self.TARGET_MELODY[self.melody_progress] for n in self.falling_notes):
                self._spawn_note()
        
        # --- 5. Termination and Final Reward ---
        self.steps += 1
        terminated = self.game_over
        truncated = False
        
        if self.melody_progress == len(self.TARGET_MELODY):
            reward += 100.0
            self.score += 100
            terminated = True

        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Game over on step limit
        
        self.last_space_held = space_held
        
        return (self._get_observation(), reward, terminated, truncated, self._get_info())

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "melody_progress": f"{self.melody_progress}/{len(self.TARGET_MELODY)}",
            "terraform_level": self.terraform_level,
        }

    def _spawn_note(self):
        if self.melody_progress >= len(self.TARGET_MELODY):
            return

        note_name = self.TARGET_MELODY[self.melody_progress]
        note = {
            'pos': [self.np_random.uniform(0.2, 0.8) * self.WIDTH, -20],
            'radius': 20,
            'name': note_name,
            'color': self.NOTE_MAP[note_name]['color']
        }
        self.falling_notes.append(note)
        self.note_spawn_timer = 120

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-max_speed, max_speed), self.np_random.uniform(-max_speed, max_speed)],
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for piece in self.scrapyard_pieces:
            pygame.draw.rect(self.screen, self.COLOR_SCRAP, piece)
        
        if self.terraform_level > 0.01:
            terraform_height = self.terraform_level * self.HEIGHT
            color1 = (20, 40, 30)
            color2 = (50, 120, 80)
            
            for y in range(int(terraform_height)):
                ratio = y / terraform_height
                color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
                pygame.draw.line(self.screen, color, (0, self.HEIGHT - y), (self.WIDTH, self.HEIGHT - y))

            for i in range(10):
                seed = i * 31337
                vine_x = (self.WIDTH / 10) * i + (seed % 10 - 5)
                max_h = terraform_height * (0.5 + (seed % 100) / 200.0)
                points = [(vine_x + math.sin(j * 0.5 + seed) * 5, self.HEIGHT - j * 10) for j in range(int(max_h / 10))]
                if len(points) > 1:
                    pygame.draw.lines(self.screen, color2, False, points, 2)

    def _render_game(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['radius'])

        for i, platform in enumerate(self.target_platforms):
            is_next = (i == self.melody_progress)
            color = self.COLOR_SUCCESS if is_next else (100, 100, 120)
            alpha = 255 if is_next else 100
            
            for j in range(10, 0, -2):
                pygame.gfxdraw.filled_circle(self.screen, int(platform['pos'][0]), int(platform['pos'][1]), platform['radius'] + j, (*color, int(alpha * (1 - j/12))))
            
            pygame.gfxdraw.filled_circle(self.screen, int(platform['pos'][0]), int(platform['pos'][1]), platform['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, int(platform['pos'][0]), int(platform['pos'][1]), platform['radius'], color)
            
            text = self.font_small.render(self.NOTE_MAP[platform['note_name']]['symbol'], True, self.COLOR_BG)
            self.screen.blit(text, text.get_rect(center=platform['pos']))

        for note in self.placed_notes:
            pygame.draw.circle(self.screen, self.NOTE_MAP[note['name']]['color'], note['pos'], note['radius'])

        for note in self.falling_notes:
            for j in range(15, 0, -3):
                pygame.gfxdraw.filled_circle(self.screen, int(note['pos'][0]), int(note['pos'][1]), note['radius'] + j, (*note['color'], int(100 * (1 - j/18))))
            pygame.draw.circle(self.screen, note['color'], note['pos'], note['radius'])
            text = self.font_small.render(note['name'], True, (0,0,0))
            self.screen.blit(text, text.get_rect(center=note['pos']))

        end_x = self.magnet_center[0] + math.cos(self.player_beam_angle) * self.HEIGHT
        end_y = self.magnet_center[1] + math.sin(self.player_beam_angle) * self.HEIGHT
        pygame.draw.line(self.screen, self.COLOR_BEAM_GLOW, self.magnet_center, (end_x, end_y), 10)
        pygame.draw.line(self.screen, self.COLOR_BEAM, self.magnet_center, (end_x, end_y), 4)

        if self.captured_note:
            note = self.captured_note
            pygame.draw.circle(self.screen, note['color'], note['pos'], note['radius'])
            text = self.font_small.render(note['name'], True, (0,0,0))
            self.screen.blit(text, text.get_rect(center=note['pos']))
            for _ in range(3):
                start = (note['pos'][0] + self.np_random.uniform(-1,1)*note['radius'], note['pos'][1] + self.np_random.uniform(-1,1)*note['radius'])
                pygame.draw.aaline(self.screen, self.COLOR_BEAM, start, self.magnet_center)

    def _render_ui(self):
        for i, note_name in enumerate(self.TARGET_MELODY):
            pos_x = self.WIDTH // 2 - (len(self.TARGET_MELODY) * 30) // 2 + i * 30
            pos_y = 20
            is_done = i < self.melody_progress
            color = self.NOTE_MAP[note_name]['color'] if is_done else (50, 50, 70)
            pygame.draw.circle(self.screen, color, (pos_x, pos_y), 10)
            if i == self.melody_progress and not self.game_over:
                 pygame.draw.circle(self.screen, (255,255,255), (pos_x, pos_y), 12, 2)

        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            text_surface = self.font_large.render("FAILURE", True, self.COLOR_FAIL)
            self.screen.blit(text_surface, text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))
        elif self.melody_progress == len(self.TARGET_MELODY):
            text_surface = self.font_large.render("SUCCESS!", True, self.COLOR_SUCCESS)
            self.screen.blit(text_surface, text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Symphony of the Scrapyard")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    action = [0, 0, 0] 
    
    print(f"\nGame: Symphony of the Scrapyard")
    print(f"Description: {GameEnv.game_description}")
    print(f"Controls: {GameEnv.user_guide}")
    print("----------------\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated or truncated:
            print(f"Game Over. Score: {info['score']}. Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False
            continue

        keys = pygame.key.get_pressed()
        action[0] = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        
    env.close()